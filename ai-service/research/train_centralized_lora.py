import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model

# ==================================================
# PATHS AND HYPERPARAMETERS
# ==================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models" / "centralized_lora"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BASE_MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value"]

ID2LABEL = {0: "Non-Stress", 1: "Stress"}
LABEL2ID = {"Non-Stress": 0, "Stress": 1}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StressDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def evaluate_model(model, dataloader, device, use_bf16: bool) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            total_loss += float(outputs.loss.item()) * len(labels)
            logits_fp32 = outputs.logits.float()
            probs = torch.softmax(logits_fp32, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_logits.append(logits_fp32.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = np.concatenate(all_preds, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)

    avg_loss = total_loss / len(labels_arr)
    acc = accuracy_score(labels_arr, preds_arr)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, average="binary", zero_division=0
    )
    cm = confusion_matrix(labels_arr, preds_arr).tolist()
    report = classification_report(
        labels_arr,
        preds_arr,
        target_names=["Non-Stress", "Stress"],
        output_dict=True,
        zero_division=0,
    )

    return {
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "logits": logits_arr,
        "probs": probs_arr,
        "preds": preds_arr,
        "labels": labels_arr,
    }


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

    print("=" * 65)
    print("PHASE 3: CENTRALIZED ROBERTA + LORA BASELINE (DREADDIT)")
    print("=" * 65)
    print(f"Device: {device} | bfloat16 enabled: {use_bf16}")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # 1. Load processed Dreaddit splits
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "validation.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Loaded splits -> Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # 2. Load tokenizer and build datasets
    print(f"\nLoading tokenizer and base model: {BASE_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    train_dataset = StressDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, MAX_LENGTH)
    val_dataset = StressDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, MAX_LENGTH)
    test_dataset = StressDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. Load RoBERTa-base and attach LoRA adapter
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = round(100.0 * trainable_params / total_params, 4)
    print(f"Parameters -> Trainable: {trainable_params:,} / Total: {total_params:,} ({trainable_pct}%)")

    # 4. Optimizer and linear warmup scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # 5. Training loop (track best model by validation F1)
    print(f"\nStarting training for {EPOCHS} epochs ({total_steps} total steps)...")
    best_val_f1 = -1.0
    best_epoch = 0
    best_state_dict = None
    epoch_history = []

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.item()) * len(labels)

        train_loss = round(running_loss / len(train_dataset), 4)
        val_eval = evaluate_model(model, val_loader, device, use_bf16)
        ep_time = round(time.time() - ep_start, 2)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_eval["loss"],
            "val_accuracy": val_eval["accuracy"],
            "val_precision": val_eval["precision"],
            "val_recall": val_eval["recall"],
            "val_f1": val_eval["f1"],
            "epoch_seconds": ep_time,
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch {epoch}/{EPOCHS} ({ep_time}s) | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_eval['loss']:.4f} | "
            f"Val Acc: {val_eval['accuracy']:.4f} | "
            f"Val F1: {val_eval['f1']:.4f}"
        )

        if val_eval["f1"] > best_val_f1:
            best_val_f1 = val_eval["f1"]
            best_epoch = epoch
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # 6. Restore best checkpoint, save adapter + tokenizer, and run final evaluations
    print(f"\nRestoring best model from Epoch {best_epoch} (Best Val F1: {best_val_f1:.4f})...")
    model.load_state_dict(best_state_dict)

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    print(f"Saved centralized LoRA adapter & tokenizer to: {MODEL_DIR}")

    final_val = evaluate_model(model, val_loader, device, use_bf16)
    final_test = evaluate_model(model, test_loader, device, use_bf16)

    # Save raw prediction probabilities/logits for Phase 6 Reliability/Calibration
    val_preds_path = EXPERIMENTS_DIR / "centralized_val_predictions.csv"
    test_preds_path = EXPERIMENTS_DIR / "centralized_test_predictions.csv"
    save_predictions_csv(val_df, final_val, val_preds_path)
    save_predictions_csv(test_df, final_test, test_preds_path)

    total_time = round(time.time() - start_time, 2)

    metrics_payload = {
        "experiment": "centralized_roberta_lora_baseline",
        "dataset": "Dreaddit",
        "base_model": BASE_MODEL_NAME,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "total_runtime_seconds": total_time,
        "hyperparameters": {
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "bfloat16": use_bf16,
        },
        "parameters": {
            "trainable_parameters": trainable_params,
            "total_parameters": total_params,
            "trainable_percentage": trainable_pct,
        },
        "best_epoch": best_epoch,
        "epoch_history": epoch_history,
        "validation_metrics": {
            "loss": final_val["loss"],
            "accuracy": final_val["accuracy"],
            "precision": final_val["precision"],
            "recall": final_val["recall"],
            "f1": final_val["f1"],
            "confusion_matrix": final_val["confusion_matrix"],
            "classification_report": final_val["classification_report"],
        },
        "test_metrics": {
            "loss": final_test["loss"],
            "accuracy": final_test["accuracy"],
            "precision": final_test["precision"],
            "recall": final_test["recall"],
            "f1": final_test["f1"],
            "confusion_matrix": final_test["confusion_matrix"],
            "classification_report": final_test["classification_report"],
        },
        "saved_artifacts": {
            "model_dir": str(MODEL_DIR),
            "val_predictions_csv": str(val_preds_path),
            "test_predictions_csv": str(test_preds_path),
        },
    }

    metrics_path = EXPERIMENTS_DIR / "centralized_baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print("\n" + "=" * 65)
    print("FINAL TEST SET EVALUATION (CENTRALIZED ROBERTA + LORA)")
    print("=" * 65)
    print(f"Test Loss:      {final_test['loss']:.4f}")
    print(f"Test Accuracy:  {final_test['accuracy']:.4f}")
    print(f"Test Precision: {final_test['precision']:.4f}")
    print(f"Test Recall:    {final_test['recall']:.4f}")
    print(f"Test F1-Score:  {final_test['f1']:.4f}")
    print(f"Confusion Matrix [rows=true (0,1), cols=pred (0,1)]: {final_test['confusion_matrix']}")
    print(f"\nTotal Runtime: {total_time} seconds")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()