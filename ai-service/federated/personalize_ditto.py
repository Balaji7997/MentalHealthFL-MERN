import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Ensure ai-service root is on sys.path when run as `python federated\personalize_ditto.py`
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from federated.client import (
    BASE_MODEL_NAME,
    ID2LABEL,
    LABEL2ID,
    FederatedClient,
    get_trainable_state_dict,
    set_trainable_state_dict,
)

# ==================================================
# PATHS AND DITTO-LORA HYPERPARAMETERS
# ==================================================
CLIENTS_DIR = BASE_DIR / "data" / "clients"
GLOBAL_MODEL_DIR = BASE_DIR / "models" / "federated_global_lora"
DITTO_MODELS_DIR = BASE_DIR / "models" / "personalized_ditto_lora"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

DITTO_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
MAX_LENGTH = 256
ADAPT_EPOCHS = 3
ADAPT_LR = 5e-5
WEIGHT_DECAY = 0.01
PROXIMAL_MU = 0.1  # Ditto L2 proximal regularization strength toward global FedAvg weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_saved_global_peft_model(device: torch.device):
    """Load the converged Phase 4 FedAvg Global RoBERTa + LoRA model as trainable."""
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = PeftModel.from_pretrained(base_model, str(GLOBAL_MODEL_DIR), is_trainable=True)
    model.to(device)
    return model


def build_predictions_df(client_id: str, df: pd.DataFrame, eval_out: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "client_id": client_id,
        "post_id": df["post_id"].values,
        "subreddit": df["subreddit"].values,
        "domain": df["domain"].values,
        "true_label": eval_out["labels"],
        "pred_label": eval_out["preds"],
        "logit_non_stress": eval_out["logits"][:, 0],
        "logit_stress": eval_out["logits"][:, 1],
        "prob_non_stress": eval_out["probs"][:, 0],
        "prob_stress": eval_out["probs"][:, 1],
    })


def personalize_client_ditto(
    model: torch.nn.Module,
    client: FederatedClient,
    global_trainable_state: Dict[str, torch.Tensor],
    epochs: int,
    lr: float,
    weight_decay: float,
    proximal_mu: float,
    device: torch.device,
    use_bf16: bool,
) -> tuple[Dict[str, torch.Tensor], dict]:
    """
    Ditto-LoRA Personalized Adaptation (Li et al., ICML 2021):
    Minimizes L_local(w_k) + (mu / 2) * ||w_k - w_global||^2
    starting from the converged FedAvg global LoRA model, selecting the best
    checkpoint on the client's local validation set (including epoch 0 = global).
    """
    set_trainable_state_dict(model, global_trainable_state)
    model.to(device)

    # Evaluate Epoch 0 (Global FedAvg baseline on this client's validation set)
    val_epoch0 = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)
    best_val_f1 = val_epoch0["f1"]
    best_val_loss = val_epoch0["loss"]
    best_epoch = 0
    best_state = {k: v.clone() for k, v in global_trainable_state.items()}

    # Keep global reference tensors on GPU for fast proximal L2 penalty
    global_ref_gpu = {
        k: v.to(device=device, dtype=torch.float32)
        for k, v in global_trainable_state.items()
    }

    trainable_named_params = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [p for _, p in trainable_named_params],
        lr=lr,
        weight_decay=weight_decay,
    )

    epoch_logs = [{
        "epoch": 0,
        "train_loss": None,
        "val_loss": val_epoch0["loss"],
        "val_accuracy": val_epoch0["accuracy"],
        "val_f1": val_epoch0["f1"],
    }]

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch in client.train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                task_loss = outputs.loss

            # Ditto proximal penalty: (mu / 2) * ||w_k - w_global||^2
            prox_term = torch.tensor(0.0, device=device, dtype=torch.float32)
            for name, param in trainable_named_params:
                diff = param.float() - global_ref_gpu[name]
                prox_term = prox_term + torch.sum(diff * diff)
            loss = task_loss.float() + 0.5 * proximal_mu * prox_term

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in trainable_named_params], 1.0)
            optimizer.step()

            batch_len = len(labels)
            total_loss += float(task_loss.item()) * batch_len
            total_items += batch_len

        avg_train_loss = round(total_loss / max(total_items, 1), 4)
        val_eval = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)

        epoch_logs.append({
            "epoch": ep,
            "train_loss": avg_train_loss,
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
            "val_f1": val_eval["f1"],
        })

        # Select checkpoint by higher Val F1 (or same F1 with lower Val Loss)
        if (val_eval["f1"] > best_val_f1) or (
            val_eval["f1"] == best_val_f1 and val_eval["loss"] < best_val_loss
        ):
            best_val_f1 = val_eval["f1"]
            best_val_loss = val_eval["loss"]
            best_epoch = ep
            best_state = get_trainable_state_dict(model)

    summary = {
        "best_adapt_epoch": best_epoch,
        "val_f1_global": val_epoch0["f1"],
        "val_f1_personalized": best_val_f1,
        "val_loss_global": val_epoch0["loss"],
        "val_loss_personalized": best_val_loss,
        "epoch_logs": epoch_logs,
    }
    return best_state, summary


def main():
    start_time = time.time()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print("=" * 78)
    print("PHASE 5.2: PERSONALIZED FEDERATED LEARNING (Ditto-LoRA Proximal Adaptation)")
    print("=" * 78)
    print(f"Device: {device} | bfloat16 enabled: {use_bf16} | Proximal mu: {PROXIMAL_MU}")

    with open(CLIENTS_DIR / "partition_summary.json", "r", encoding="utf-8") as f:
        partition_meta = json.load(f)

    with open(EXPERIMENTS_DIR / "federated_fedavg_metrics.json", "r", encoding="utf-8") as f:
        fedavg_metrics = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(str(GLOBAL_MODEL_DIR))
    model = load_saved_global_peft_model(device)
    global_trainable_state = get_trainable_state_dict(model)

    clients: List[FederatedClient] = []
    for c_info in partition_meta["clients"]:
        cid = c_info["client_id"]
        clients.append(
            FederatedClient(
                client_id=cid,
                domain=c_info["domain"],
                client_dir=CLIENTS_DIR / cid,
                tokenizer=tokenizer,
                max_length=MAX_LENGTH,
                batch_size=BATCH_SIZE,
            )
        )

    per_client_test = {}
    adaptation_logs = {}
    comparison_table = []
    all_val_pred_dfs = []
    all_test_pred_dfs = []

    print(f"\nPersonalizing each of the {len(clients)} clients (max {ADAPT_EPOCHS} epochs, lr={ADAPT_LR})...\n")

    for client in clients:
        cid = client.client_id
        best_state, adapt_summary = personalize_client_ditto(
            model=model,
            client=client,
            global_trainable_state=global_trainable_state,
            epochs=ADAPT_EPOCHS,
            lr=ADAPT_LR,
            weight_decay=WEIGHT_DECAY,
            proximal_mu=PROXIMAL_MU,
            device=device,
            use_bf16=use_bf16,
        )
        adaptation_logs[cid] = adapt_summary

        set_trainable_state_dict(model, best_state)

        # Save personalized Ditto-LoRA model for this client
        client_model_dir = DITTO_MODELS_DIR / cid
        client_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(client_model_dir)
        tokenizer.save_pretrained(client_model_dir)

        c_val = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)
        c_test = client.evaluate(model, split="test", device=device, use_bf16=use_bf16)

        all_val_pred_dfs.append(build_predictions_df(cid, client.val_df, c_val))
        all_test_pred_dfs.append(build_predictions_df(cid, client.test_df, c_test))

        global_c = fedavg_metrics["per_client_test_metrics"][cid]
        acc_delta = round(c_test["accuracy"] - global_c["accuracy"], 4)
        f1_delta = round(c_test["f1"] - global_c["f1"], 4)

        per_client_test[cid] = {
            "domain": client.domain,
            "num_samples": c_test["num_samples"],
            "loss": c_test["loss"],
            "accuracy": c_test["accuracy"],
            "precision": c_test["precision"],
            "recall": c_test["recall"],
            "f1": c_test["f1"],
            "confusion_matrix": c_test["confusion_matrix"],
            "best_adapt_epoch": adapt_summary["best_adapt_epoch"],
            "model_dir": str(client_model_dir),
        }

        comparison_table.append({
            "client_id": cid,
            "domain": client.domain,
            "test_samples": c_test["num_samples"],
            "best_adapt_epoch": adapt_summary["best_adapt_epoch"],
            "fedavg_global_accuracy": global_c["accuracy"],
            "ditto_personalized_accuracy": c_test["accuracy"],
            "accuracy_delta": acc_delta,
            "fedavg_global_f1": global_c["f1"],
            "ditto_personalized_f1": c_test["f1"],
            "f1_delta": f1_delta,
        })

        print(
            f"  {cid} ({client.domain:<9}, best_ep={adapt_summary['best_adapt_epoch']}) | "
            f"Val F1: {adapt_summary['val_f1_global']:.4f} -> {adapt_summary['val_f1_personalized']:.4f} | "
            f"Test Acc: {global_c['accuracy']:.4f} -> {c_test['accuracy']:.4f} ({acc_delta:+.4f}) | "
            f"Test F1: {global_c['f1']:.4f} -> {c_test['f1']:.4f} ({f1_delta:+.4f})"
        )

    combined_val_df = pd.concat(all_val_pred_dfs, ignore_index=True)
    combined_test_df = pd.concat(all_test_pred_dfs, ignore_index=True)

    val_preds_path = EXPERIMENTS_DIR / "personalized_ditto_val_predictions.csv"
    test_preds_path = EXPERIMENTS_DIR / "personalized_ditto_test_predictions.csv"
    combined_val_df.to_csv(val_preds_path, index=False, encoding="utf-8")
    combined_test_df.to_csv(test_preds_path, index=False, encoding="utf-8")

    y_true = combined_test_df["true_label"].values
    y_pred = combined_test_df["pred_label"].values
    overall_acc = round(float(accuracy_score(y_true, y_pred)), 4)
    overall_prec, overall_rec, overall_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    overall_cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    mean_client_acc = round(float(np.mean([v["accuracy"] for v in per_client_test.values()])), 4)
    mean_client_f1 = round(float(np.mean([v["f1"] for v in per_client_test.values()])), 4)

    total_time = round(time.time() - start_time, 2)

    metrics_payload = {
        "experiment": "personalized_federated_learning_ditto_lora",
        "method_name": "Ditto-LoRA (FedAvg Global Initialization + Proximal Regularized Local Adaptation)",
        "citation": "Li, T., Hu, S., Beirami, A., & Smith, V. (2021). Ditto: Fair and Robust Federated Learning Through Personalization. ICML 2021.",
        "dataset": "Dreaddit (5 Natural Non-IID Domain Clients)",
        "base_model": BASE_MODEL_NAME,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_runtime_seconds": total_time,
        "hyperparameters": {
            "seed": SEED,
            "adapt_epochs": ADAPT_EPOCHS,
            "adapt_lr": ADAPT_LR,
            "weight_decay": WEIGHT_DECAY,
            "proximal_mu": PROXIMAL_MU,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "bfloat16": use_bf16,
        },
        "adaptation_logs": adaptation_logs,
        "overall_personalized_test_metrics": {
            "num_samples": int(len(y_true)),
            "accuracy": overall_acc,
            "precision": round(float(overall_prec), 4),
            "recall": round(float(overall_rec), 4),
            "f1": round(float(overall_f1), 4),
            "confusion_matrix": overall_cm,
        },
        "mean_client_test_metrics": {
            "accuracy": mean_client_acc,
            "f1": mean_client_f1,
        },
        "per_client_test_metrics": per_client_test,
        "comparison_vs_fedavg_global": comparison_table,
        "saved_artifacts": {
            "models_root_dir": str(DITTO_MODELS_DIR),
            "val_predictions_csv": str(val_preds_path),
            "test_predictions_csv": str(test_preds_path),
        },
    }

    metrics_path = EXPERIMENTS_DIR / "personalized_ditto_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    global_overall = fedavg_metrics["final_global_test_metrics"]
    print("\n" + "=" * 78)
    print("OVERALL TEST SUMMARY (FedAvg Global vs. Ditto-LoRA Personalized)")
    print("=" * 78)
    print(
        f"FedAvg Global   | Acc: {global_overall['accuracy']:.4f} | "
        f"Prec: {global_overall['precision']:.4f} | Rec: {global_overall['recall']:.4f} | "
        f"F1: {global_overall['f1']:.4f} | Mean Client F1: {fedavg_metrics['mean_client_test_metrics']['f1']:.4f}"
    )
    print(
        f"Ditto-LoRA Pers | Acc: {overall_acc:.4f} | "
        f"Prec: {float(overall_prec):.4f} | Rec: {float(overall_rec):.4f} | "
        f"F1: {float(overall_f1):.4f} | Mean Client F1: {mean_client_f1:.4f}"
    )
    print(f"Ditto-LoRA Confusion Matrix: {overall_cm}")
    print(f"\nTotal Runtime: {total_time} seconds")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()