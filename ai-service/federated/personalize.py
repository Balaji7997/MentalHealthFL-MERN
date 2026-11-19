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
from transformers import AutoTokenizer

# Ensure ai-service root is on sys.path when run as `python federated\personalize.py`
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from federated.client import (
    BASE_MODEL_NAME,
    FederatedClient,
    build_roberta_lora_model,
    get_trainable_state_dict,
    set_trainable_state_dict,
    split_trainable_keys,
)
from federated.fedavg import compute_state_dict_stats, federated_averaging

# ==================================================
# PATHS AND PERSONALIZED FL HYPERPARAMETERS
# ==================================================
CLIENTS_DIR = BASE_DIR / "data" / "clients"
PERSONALIZED_MODELS_DIR = BASE_DIR / "models" / "personalized_lora"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

PERSONALIZED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
HEAD_ALIGN_EPOCHS = 2
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-4
HEAD_ALIGN_LR = 1e-4
WEIGHT_DECAY = 0.01

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def align_client_head(
    model: torch.nn.Module,
    client: FederatedClient,
    shared_lora_state: Dict[str, torch.Tensor],
    private_head_state: Dict[str, torch.Tensor],
    head_keys: List[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    use_bf16: bool,
) -> Dict[str, torch.Tensor]:
    """
    Fine-tune ONLY this client's private classification head while keeping
    the shared global LoRA encoder fixed (FedRep/FedPer head alignment).
    """
    combined_state = {**shared_lora_state, **private_head_state}
    set_trainable_state_dict(model, combined_state)
    model.to(device)
    model.train()

    head_key_set = set(head_keys)
    head_params = [p for name, p in model.named_parameters() if name in head_key_set]
    optimizer = torch.optim.AdamW(head_params, lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
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
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params, 1.0)
            optimizer.step()

    updated_full = get_trainable_state_dict(model)
    return {k: updated_full[k] for k in head_keys}


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


def main():
    start_time = time.time()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print("=" * 76)
    print("PHASE 5: PERSONALIZED FEDERATED LEARNING (FedPer-LoRA)")
    print("Shared: LoRA Attention Adapters (FedAvg) | Private: Client Classifier Heads")
    print("=" * 76)
    print(f"Device: {device} | bfloat16 enabled: {use_bf16}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # 1. Load partition metadata and Phase 4 FedAvg baseline results for direct comparison
    with open(CLIENTS_DIR / "partition_summary.json", "r", encoding="utf-8") as f:
        partition_meta = json.load(f)

    fedavg_metrics_path = EXPERIMENTS_DIR / "federated_fedavg_metrics.json"
    with open(fedavg_metrics_path, "r", encoding="utf-8") as f:
        fedavg_metrics = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

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

    # 2. Initialize RoBERTa + LoRA and separate shared LoRA keys vs private head keys
    model = build_roberta_lora_model(
        base_model_name=BASE_MODEL_NAME,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model.to(device)

    init_trainable = get_trainable_state_dict(model)
    lora_keys, head_keys = split_trainable_keys(init_trainable)

    global_lora_state = {k: init_trainable[k].clone() for k in lora_keys}
    client_head_states = {
        c.client_id: {k: init_trainable[k].clone() for k in head_keys}
        for c in clients
    }

    shared_stats = compute_state_dict_stats(global_lora_state)
    private_stats = compute_state_dict_stats(client_head_states[clients[0].client_id])

    print(
        f"\nShared (Federated) LoRA params:  {shared_stats['num_parameters']:,} "
        f"({shared_stats['size_mb']:.2f} MB across {shared_stats['num_tensors']} tensors)"
    )
    print(
        f"Private (Local) Head params:     {private_stats['num_parameters']:,} "
        f"({private_stats['size_mb']:.2f} MB across {private_stats['num_tensors']} tensors)"
    )

    # 3. Multi-round FedPer-LoRA training
    rounds_history = []
    best_client_val_f1 = {c.client_id: -1.0 for c in clients}
    best_client_round = {c.client_id: 0 for c in clients}
    best_client_states = {}

    print(f"\nStarting {NUM_ROUNDS} FedPer-LoRA Rounds ({LOCAL_EPOCHS} local epoch/round, lr={LEARNING_RATE})...\n")

    for rnd in range(1, NUM_ROUNDS + 1):
        rnd_start = time.time()
        print(f"--- FedPer-LoRA Round {rnd}/{NUM_ROUNDS} ---")

        lora_updates = []
        round_client_logs = []

        # Step A: Local training (client merges shared LoRA + its own private head)
        for client in clients:
            c_start = time.time()
            cid = client.client_id
            client_init_state = {**global_lora_state, **client_head_states[cid]}

            updated_state, n_samples, local_loss = client.train_local(
                model=model,
                init_trainable_state=client_init_state,
                local_epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                device=device,
                use_bf16=use_bf16,
            )

            # Separate updates: upload ONLY LoRA weights; keep classifier head PRIVATE
            client_lora_update = {k: updated_state[k] for k in lora_keys}
            client_head_states[cid] = {k: updated_state[k] for k in head_keys}
            lora_updates.append((client_lora_update, n_samples))

            c_time = round(time.time() - c_start, 2)
            round_client_logs.append({
                "client_id": cid,
                "domain": client.domain,
                "num_train_samples": n_samples,
                "local_train_loss": local_loss,
                "train_seconds": c_time,
            })

        # Step B: Server aggregates ONLY shared LoRA tensors via weighted FedAvg
        global_lora_state = federated_averaging(lora_updates, keys_to_aggregate=lora_keys)

        # Step C: Evaluate each client's personalized model (new shared LoRA + private head) on its local val set
        val_accs = []
        val_f1s = []
        for log_entry, client in zip(round_client_logs, clients):
            cid = client.client_id
            personalized_state = {**global_lora_state, **client_head_states[cid]}
            set_trainable_state_dict(model, personalized_state)
            c_val = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)

            log_entry["personalized_val_loss"] = c_val["loss"]
            log_entry["personalized_val_accuracy"] = c_val["accuracy"]
            log_entry["personalized_val_f1"] = c_val["f1"]

            val_accs.append(c_val["accuracy"])
            val_f1s.append(c_val["f1"])

            if c_val["f1"] > best_client_val_f1[cid]:
                best_client_val_f1[cid] = c_val["f1"]
                best_client_round[cid] = rnd
                best_client_states[cid] = {k: v.clone() for k, v in personalized_state.items()}

            print(
                f"  {cid} ({client.domain:<9}, n={client.num_train_samples:>3}) | "
                f"Train Loss: {log_entry['local_train_loss']:.4f} | "
                f"Pers. Val Acc: {c_val['accuracy']:.4f} | "
                f"Pers. Val F1: {c_val['f1']:.4f}"
            )

        mean_val_acc = round(float(np.mean(val_accs)), 4)
        mean_val_f1 = round(float(np.mean(val_f1s)), 4)
        rnd_time = round(time.time() - rnd_start, 2)

        print(
            f"  >> FedPer-LoRA Mean Client Val (Round {rnd}) | "
            f"Mean Acc: {mean_val_acc:.4f} | Mean F1: {mean_val_f1:.4f} | Time: {rnd_time}s\n"
        )

        rounds_history.append({
            "round": rnd,
            "clients": round_client_logs,
            "mean_client_val_accuracy": mean_val_acc,
            "mean_client_val_f1": mean_val_f1,
            "round_seconds": rnd_time,
        })

    # 4. Final Local Head Alignment (with frozen shared LoRA) — keep only if Val F1 improves
    print("=" * 76)
    print("Running Local Private Head Alignment (Frozen Shared LoRA, Val-Selected)...")
    head_alignment_log = {}

    for client in clients:
        cid = client.client_id
        best_state = best_client_states[cid]
        base_lora = {k: best_state[k] for k in lora_keys}
        base_head = {k: best_state[k] for k in head_keys}
        before_f1 = best_client_val_f1[cid]

        aligned_head = align_client_head(
            model=model,
            client=client,
            shared_lora_state=base_lora,
            private_head_state=base_head,
            head_keys=head_keys,
            epochs=HEAD_ALIGN_EPOCHS,
            lr=HEAD_ALIGN_LR,
            weight_decay=WEIGHT_DECAY,
            device=device,
            use_bf16=use_bf16,
        )

        candidate_state = {**base_lora, **aligned_head}
        set_trainable_state_dict(model, candidate_state)
        aligned_val = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)

        accepted = aligned_val["f1"] > before_f1
        if accepted:
            best_client_states[cid] = {k: v.clone() for k, v in candidate_state.items()}
            best_client_val_f1[cid] = aligned_val["f1"]

        head_alignment_log[cid] = {
            "best_fedper_round": best_client_round[cid],
            "val_f1_before_alignment": before_f1,
            "val_f1_after_alignment": aligned_val["f1"],
            "alignment_accepted": accepted,
        }
        status = "ACCEPTED" if accepted else "kept FedPer checkpoint"
        print(
            f"  {cid} ({client.domain:<9}) | Best Round: {best_client_round[cid]} | "
            f"Val F1: {before_f1:.4f} -> {aligned_val['f1']:.4f} ({status})"
        )

    # 5. Save each client's personalized model & evaluate on untouched client test splits
    print("\n" + "=" * 76)
    print("FINAL PER-CLIENT TEST EVALUATION: FEDAVG GLOBAL vs. PERSONALIZED (FedPer-LoRA)")
    print("=" * 76)

    per_client_test = {}
    comparison_table = []
    all_val_pred_dfs = []
    all_test_pred_dfs = []

    for client in clients:
        cid = client.client_id
        set_trainable_state_dict(model, best_client_states[cid])

        # Save each client's personalized adapter + tokenizer
        client_model_dir = PERSONALIZED_MODELS_DIR / cid
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
            "model_dir": str(client_model_dir),
        }

        comparison_table.append({
            "client_id": cid,
            "domain": client.domain,
            "test_samples": c_test["num_samples"],
            "fedavg_global_accuracy": global_c["accuracy"],
            "personalized_accuracy": c_test["accuracy"],
            "accuracy_delta": acc_delta,
            "fedavg_global_f1": global_c["f1"],
            "personalized_f1": c_test["f1"],
            "f1_delta": f1_delta,
        })

        print(
            f"  {cid} ({client.domain:<9}, n={c_test['num_samples']:>3}) | "
            f"Acc: {global_c['accuracy']:.4f} -> {c_test['accuracy']:.4f} ({acc_delta:+.4f}) | "
            f"F1: {global_c['f1']:.4f} -> {c_test['f1']:.4f} ({f1_delta:+.4f})"
        )

    # Combine all clients' validation and test predictions into unified CSVs for Phase 6
    combined_val_df = pd.concat(all_val_pred_dfs, ignore_index=True)
    combined_test_df = pd.concat(all_test_pred_dfs, ignore_index=True)

    val_preds_path = EXPERIMENTS_DIR / "personalized_val_predictions.csv"
    test_preds_path = EXPERIMENTS_DIR / "personalized_test_predictions.csv"
    combined_val_df.to_csv(val_preds_path, index=False, encoding="utf-8")
    combined_test_df.to_csv(test_preds_path, index=False, encoding="utf-8")

    # Compute overall pooled test metrics (evaluating each test sample with its domain's personalized client model)
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
        "experiment": "personalized_federated_learning_fedper_lora",
        "method_name": "FedPer-LoRA (Shared LoRA Encoder + Private Personalized Classifier Head)",
        "dataset": "Dreaddit (5 Natural Non-IID Domain Clients)",
        "base_model": BASE_MODEL_NAME,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_runtime_seconds": total_time,
        "parameter_split": {
            "shared_parameters_name": "RoBERTa Attention LoRA Adapters (query, key, value)",
            "shared_tensors": shared_stats["num_tensors"],
            "shared_parameters": shared_stats["num_parameters"],
            "shared_upload_mb_per_client_round": shared_stats["size_mb"],
            "personalized_parameters_name": "Sequence Classification Head (dense + out_proj)",
            "personalized_tensors": private_stats["num_tensors"],
            "personalized_parameters": private_stats["num_parameters"],
            "personalized_local_mb_per_client": private_stats["size_mb"],
            "communication_reduction_vs_fedavg_percent": round(
                100.0 * (1.0 - shared_stats["size_mb"] / fedavg_metrics["communication_efficiency"]["upload_size_per_client_mb"]),
                2,
            ),
        },
        "hyperparameters": {
            "seed": SEED,
            "num_clients": len(clients),
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "head_align_epochs": HEAD_ALIGN_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "head_align_lr": HEAD_ALIGN_LR,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "bfloat16": use_bf16,
        },
        "rounds_history": rounds_history,
        "head_alignment_log": head_alignment_log,
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
            "models_root_dir": str(PERSONALIZED_MODELS_DIR),
            "val_predictions_csv": str(val_preds_path),
            "test_predictions_csv": str(test_preds_path),
        },
    }

    metrics_path = EXPERIMENTS_DIR / "personalized_fedper_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    global_overall = fedavg_metrics["final_global_test_metrics"]
    print("\n" + "=" * 76)
    print("OVERALL TEST SUMMARY (FedAvg Global vs. Personalized FedPer-LoRA)")
    print("=" * 76)
    print(
        f"FedAvg Global        | Acc: {global_overall['accuracy']:.4f} | "
        f"Prec: {global_overall['precision']:.4f} | Rec: {global_overall['recall']:.4f} | "
        f"F1: {global_overall['f1']:.4f} | Mean Client F1: {fedavg_metrics['mean_client_test_metrics']['f1']:.4f}"
    )
    print(
        f"Personalized FedPer  | Acc: {overall_acc:.4f} | "
        f"Prec: {float(overall_prec):.4f} | Rec: {float(overall_rec):.4f} | "
        f"F1: {float(overall_f1):.4f} | Mean Client F1: {mean_client_f1:.4f}"
    )
    print(f"Personalized Confusion Matrix: {overall_cm}")
    print(
        f"Communication per client/round: {fedavg_metrics['communication_efficiency']['upload_size_per_client_mb']:.2f} MB (FedAvg) "
        f"-> {shared_stats['size_mb']:.2f} MB (FedPer-LoRA)"
    )
    print(f"\nTotal Runtime: {total_time} seconds")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()