import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure ai-service root is on sys.path when run as `python federated\server.py`
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from federated.client import (
    BASE_MODEL_NAME,
    FederatedClient,
    StressDataset,
    build_roberta_lora_model,
    evaluate_dataloader,
    get_trainable_state_dict,
    set_trainable_state_dict,
)
from federated.fedavg import compute_state_dict_stats, federated_averaging

# ==================================================
# PATHS AND FEDERATED HYPERPARAMETERS
# ==================================================
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLIENTS_DIR = BASE_DIR / "data" / "clients"
MODEL_DIR = BASE_DIR / "models" / "federated_global_lora"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 16
MAX_LENGTH = 256
LEARNING_RATE = 2e-4
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


def save_predictions_csv(df: pd.DataFrame, eval_out: dict, path: Path) -> None:
    out_df = pd.DataFrame({
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
    out_df.to_csv(path, index=False, encoding="utf-8")


def main():
    start_time = time.time()
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    print("=" * 72)
    print("PHASE 4.4: FEDERATED LEARNING WITH FEDAVG (5 NON-IID CLIENTS)")
    print("=" * 72)
    print(f"Device: {device} | bfloat16 enabled: {use_bf16}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # 1. Load partition metadata and global validation/test splits
    with open(CLIENTS_DIR / "partition_summary.json", "r", encoding="utf-8") as f:
        partition_meta = json.load(f)

    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    print(f"\nLoading tokenizer & initializing shared RoBERTa + LoRA model ({BASE_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    global_val_loader = DataLoader(
        StressDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    global_test_loader = DataLoader(
        StressDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # 2. Initialize all 5 federated clients
    clients = []
    for c_info in partition_meta["clients"]:
        cid = c_info["client_id"]
        domain = c_info["domain"]
        client = FederatedClient(
            client_id=cid,
            domain=domain,
            client_dir=CLIENTS_DIR / cid,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            batch_size=BATCH_SIZE,
        )
        clients.append(client)

    print(f"Initialized {len(clients)} clients: {[f'{c.client_id}({c.domain}:{c.num_train_samples})' for c in clients]}")

    # 3. Initialize single reusable RoBERTa + LoRA model on GPU
    model = build_roberta_lora_model(
        base_model_name=BASE_MODEL_NAME,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model.to(device)

    global_trainable_state = get_trainable_state_dict(model)
    comm_stats = compute_state_dict_stats(global_trainable_state)
    print(
        f"Communicated payload per client/round: {comm_stats['num_parameters']:,} params "
        f"({comm_stats['size_mb']:.2f} MB across {comm_stats['num_tensors']} tensors)"
    )

    # 4. Multi-round FedAvg training loop
    best_val_f1 = -1.0
    best_round = 0
    best_global_state = None
    rounds_history = []

    print(f"\nStarting {NUM_ROUNDS} Federated Rounds ({LOCAL_EPOCHS} local epoch/round, lr={LEARNING_RATE})...\n")

    for rnd in range(1, NUM_ROUNDS + 1):
        rnd_start = time.time()
        print(f"--- Round {rnd}/{NUM_ROUNDS} ---")

        client_updates = []
        client_round_logs = []

        # Each client trains locally starting from the current global trainable state
        for client in clients:
            c_start = time.time()
            updated_state, n_samples, local_loss = client.train_local(
                model=model,
                init_trainable_state=global_trainable_state,
                local_epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                device=device,
                use_bf16=use_bf16,
            )
            # Evaluate this client's local model on its own validation split before aggregation
            local_val = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)
            c_time = round(time.time() - c_start, 2)

            client_updates.append((updated_state, n_samples))
            client_round_logs.append({
                "client_id": client.client_id,
                "domain": client.domain,
                "num_train_samples": n_samples,
                "local_train_loss": local_loss,
                "local_val_accuracy": local_val["accuracy"],
                "local_val_f1": local_val["f1"],
                "train_seconds": c_time,
            })

            print(
                f"  {client.client_id} ({client.domain:<9}, n={n_samples:>3}) | "
                f"Train Loss: {local_loss:.4f} | Local Val Acc: {local_val['accuracy']:.4f} | "
                f"Local Val F1: {local_val['f1']:.4f} ({c_time}s)"
            )

        # Server executes weighted FedAvg aggregation
        total_n = sum(n for _, n in client_updates)
        aggregation_weights = {
            clients[i].client_id: round(client_updates[i][1] / total_n, 4)
            for i in range(len(clients))
        }
        global_trainable_state = federated_averaging(client_updates)
        set_trainable_state_dict(model, global_trainable_state)

        # Evaluate aggregated global model on overall validation and each client's validation
        global_val = evaluate_dataloader(model, global_val_loader, device=device, use_bf16=use_bf16)

        client_global_val_metrics = {}
        for client in clients:
            c_gval = client.evaluate(model, split="val", device=device, use_bf16=use_bf16)
            client_global_val_metrics[client.client_id] = {
                "domain": client.domain,
                "loss": c_gval["loss"],
                "accuracy": c_gval["accuracy"],
                "precision": c_gval["precision"],
                "recall": c_gval["recall"],
                "f1": c_gval["f1"],
            }

        rnd_time = round(time.time() - rnd_start, 2)
        print(
            f"  >> FedAvg Global Model (Round {rnd}) | "
            f"Val Loss: {global_val['loss']:.4f} | Val Acc: {global_val['accuracy']:.4f} | "
            f"Val F1: {global_val['f1']:.4f} | Round Time: {rnd_time}s\n"
        )

        rounds_history.append({
            "round": rnd,
            "aggregation_weights": aggregation_weights,
            "clients": client_round_logs,
            "global_val_metrics": {
                "loss": global_val["loss"],
                "accuracy": global_val["accuracy"],
                "precision": global_val["precision"],
                "recall": global_val["recall"],
                "f1": global_val["f1"],
                "confusion_matrix": global_val["confusion_matrix"],
            },
            "global_model_on_clients_val": client_global_val_metrics,
            "round_seconds": rnd_time,
        })

        if global_val["f1"] > best_val_f1:
            best_val_f1 = global_val["f1"]
            best_round = rnd
            best_global_state = {k: v.clone() for k, v in global_trainable_state.items()}

    # 5. Restore best global round, save adapter + tokenizer, and run final test evaluations
    print("=" * 72)
    print(f"Restoring Best FedAvg Global Model from Round {best_round} (Best Val F1: {best_val_f1:.4f})...")
    set_trainable_state_dict(model, best_global_state)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved FedAvg global LoRA adapter & tokenizer to: {MODEL_DIR}")

    final_global_val = evaluate_dataloader(model, global_val_loader, device=device, use_bf16=use_bf16)
    final_global_test = evaluate_dataloader(model, global_test_loader, device=device, use_bf16=use_bf16)

    # Per-client test evaluation of the global FedAvg model (important baseline for Phase 5!)
    per_client_test = {}
    print("\nGlobal FedAvg Model Performance on Each Client's Test Set:")
    for client in clients:
        c_test = client.evaluate(model, split="test", device=device, use_bf16=use_bf16)
        per_client_test[client.client_id] = {
            "domain": client.domain,
            "num_samples": c_test["num_samples"],
            "loss": c_test["loss"],
            "accuracy": c_test["accuracy"],
            "precision": c_test["precision"],
            "recall": c_test["recall"],
            "f1": c_test["f1"],
            "confusion_matrix": c_test["confusion_matrix"],
        }
        print(
            f"  {client.client_id} ({client.domain:<9}, n={c_test['num_samples']:>3}) | "
            f"Acc: {c_test['accuracy']:.4f} | Prec: {c_test['precision']:.4f} | "
            f"Rec: {c_test['recall']:.4f} | F1: {c_test['f1']:.4f}"
        )

    mean_client_test_acc = round(float(np.mean([v["accuracy"] for v in per_client_test.values()])), 4)
    mean_client_test_f1 = round(float(np.mean([v["f1"] for v in per_client_test.values()])), 4)

    # Save raw predictions for Phase 6 Reliability/Calibration
    val_preds_path = EXPERIMENTS_DIR / "federated_global_val_predictions.csv"
    test_preds_path = EXPERIMENTS_DIR / "federated_global_test_predictions.csv"
    save_predictions_csv(val_df, final_global_val, val_preds_path)
    save_predictions_csv(test_df, final_global_test, test_preds_path)

    total_time = round(time.time() - start_time, 2)

    metrics_payload = {
        "experiment": "federated_fedavg_roberta_lora",
        "dataset": "Dreaddit (5 Natural Non-IID Domain Clients)",
        "base_model": BASE_MODEL_NAME,
        "aggregation_algorithm": "FedAvg (Sample-Weighted Parameter Averaging on Trainable LoRA + Classifier)",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_runtime_seconds": total_time,
        "hyperparameters": {
            "seed": SEED,
            "num_clients": len(clients),
            "num_rounds": NUM_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "bfloat16": use_bf16,
        },
        "communication_efficiency": {
            "shared_tensors_per_round": comm_stats["num_tensors"],
            "shared_parameters_per_round": comm_stats["num_parameters"],
            "upload_size_per_client_mb": comm_stats["size_mb"],
            "total_upload_all_rounds_mb": round(comm_stats["size_mb"] * len(clients) * NUM_ROUNDS, 4),
        },
        "best_round": best_round,
        "rounds_history": rounds_history,
        "final_global_val_metrics": {
            "loss": final_global_val["loss"],
            "accuracy": final_global_val["accuracy"],
            "precision": final_global_val["precision"],
            "recall": final_global_val["recall"],
            "f1": final_global_val["f1"],
            "confusion_matrix": final_global_val["confusion_matrix"],
        },
        "final_global_test_metrics": {
            "loss": final_global_test["loss"],
            "accuracy": final_global_test["accuracy"],
            "precision": final_global_test["precision"],
            "recall": final_global_test["recall"],
            "f1": final_global_test["f1"],
            "confusion_matrix": final_global_test["confusion_matrix"],
        },
        "per_client_test_metrics": per_client_test,
        "mean_client_test_metrics": {
            "accuracy": mean_client_test_acc,
            "f1": mean_client_test_f1,
        },
        "saved_artifacts": {
            "model_dir": str(MODEL_DIR),
            "val_predictions_csv": str(val_preds_path),
            "test_predictions_csv": str(test_preds_path),
        },
    }

    metrics_path = EXPERIMENTS_DIR / "federated_fedavg_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 72)
    print("FINAL OVERALL TEST EVALUATION (FEDAVG GLOBAL MODEL)")
    print("=" * 72)
    print(f"Best Round:      {best_round}/{NUM_ROUNDS}")
    print(f"Test Loss:       {final_global_test['loss']:.4f}")
    print(f"Test Accuracy:   {final_global_test['accuracy']:.4f}")
    print(f"Test Precision:  {final_global_test['precision']:.4f}")
    print(f"Test Recall:     {final_global_test['recall']:.4f}")
    print(f"Test F1-Score:   {final_global_test['f1']:.4f}")
    print(f"Mean Client Acc: {mean_client_test_acc:.4f} | Mean Client F1: {mean_client_test_f1:.4f}")
    print(f"Confusion Matrix: {final_global_test['confusion_matrix']}")
    print(f"\nTotal Runtime: {total_time} seconds")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()