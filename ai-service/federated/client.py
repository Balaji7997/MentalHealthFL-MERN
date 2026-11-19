from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType, get_peft_model

BASE_MODEL_NAME = "FacebookAI/roberta-base"
ID2LABEL = {0: "Non-Stress", 1: "Stress"}
LABEL2ID = {"Non-Stress": 0, "Stress": 1}


class StressDataset(Dataset):
    """Tokenized binary stress detection dataset."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
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


def build_roberta_lora_model(
    base_model_name: str = BASE_MODEL_NAME,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
):
    """Initialize RoBERTa-base with a binary classification head and LoRA adapters."""
    if target_modules is None:
        target_modules = ["query", "key", "value"]

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    return get_peft_model(base_model, lora_config)


def get_trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only trainable parameters (LoRA adapters + classifier head) onto CPU."""
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    state = model.state_dict()
    return {
        name: state[name].detach().cpu().clone()
        for name in state
        if name in trainable_names
    }


def set_trainable_state_dict(
    model: torch.nn.Module,
    trainable_state: Dict[str, torch.Tensor],
) -> None:
    """Load trainable parameter weights into the model without touching frozen RoBERTa layers."""
    model.load_state_dict(trainable_state, strict=False)


def split_trainable_keys(trainable_state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """
    Separate trainable keys into:
      - lora_keys: shared LoRA adapter matrices in the transformer encoder
      - head_keys: classification head parameters (used for personalization in Phase 5)
    """
    lora_keys = [k for k in trainable_state.keys() if "lora_" in k]
    head_keys = [k for k in trainable_state.keys() if "lora_" not in k]
    return lora_keys, head_keys


def evaluate_dataloader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_bf16: bool = False,
) -> dict:
    """Evaluate a model on any DataLoader and return metrics plus raw logits/probabilities."""
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
    cm = confusion_matrix(labels_arr, preds_arr, labels=[0, 1]).tolist()

    return {
        "num_samples": int(len(labels_arr)),
        "loss": round(float(avg_loss), 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
        "logits": logits_arr,
        "probs": probs_arr,
        "preds": preds_arr,
        "labels": labels_arr,
    }


class FederatedClient:
    """Represents a single simulated federated client with private local Dreaddit domain data."""

    def __init__(
        self,
        client_id: str,
        domain: str,
        client_dir: Path,
        tokenizer,
        max_length: int = 256,
        batch_size: int = 16,
    ):
        self.client_id = client_id
        self.domain = domain
        self.client_dir = Path(client_dir)
        self.batch_size = batch_size

        self.train_df = pd.read_csv(self.client_dir / "train.csv")
        self.val_df = pd.read_csv(self.client_dir / "validation.csv")
        self.test_df = pd.read_csv(self.client_dir / "test.csv")

        self.train_dataset = StressDataset(
            self.train_df["text"].tolist(),
            self.train_df["label"].tolist(),
            tokenizer,
            max_length=max_length,
        )
        self.val_dataset = StressDataset(
            self.val_df["text"].tolist(),
            self.val_df["label"].tolist(),
            tokenizer,
            max_length=max_length,
        )
        self.test_dataset = StressDataset(
            self.test_df["text"].tolist(),
            self.test_df["label"].tolist(),
            tokenizer,
            max_length=max_length,
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    @property
    def num_train_samples(self) -> int:
        return len(self.train_dataset)

    def train_local(
        self,
        model: torch.nn.Module,
        init_trainable_state: Dict[str, torch.Tensor],
        local_epochs: int = 1,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        device: Optional[torch.device] = None,
        use_bf16: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """
        Load starting trainable weights, train on this client's private local data,
        and return (updated_trainable_state, num_train_samples, avg_train_loss).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        set_trainable_state_dict(model, init_trainable_state)
        model.to(device)
        model.train()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_loss = 0.0
        total_items = 0

        for _ in range(local_epochs):
            for batch in self.train_loader:
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

                batch_len = len(labels)
                total_loss += float(loss.item()) * batch_len
                total_items += batch_len

        avg_loss = round(total_loss / max(total_items, 1), 4)
        updated_state = get_trainable_state_dict(model)
        return updated_state, self.num_train_samples, avg_loss

    def evaluate(
        self,
        model: torch.nn.Module,
        split: str = "val",
        device: Optional[torch.device] = None,
        use_bf16: bool = False,
    ) -> dict:
        """Evaluate the given model on this client's local validation or test split."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loader = self.val_loader if split == "val" else self.test_loader
        return evaluate_dataloader(model, loader, device=device, use_bf16=use_bf16)


def _run_self_test() -> None:
    """Quick sanity check: load client_1 and inspect trainable LoRA vs classifier head keys."""
    base_dir = Path(__file__).resolve().parent.parent
    client_1_dir = base_dir / "data" / "clients" / "client_1"

    print("Running FederatedClient self-test...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    client = FederatedClient("client_1", "abuse", client_1_dir, tokenizer)
    model = build_roberta_lora_model()

    trainable_state = get_trainable_state_dict(model)
    lora_keys, head_keys = split_trainable_keys(trainable_state)

    lora_params = sum(trainable_state[k].numel() for k in lora_keys)
    head_params = sum(trainable_state[k].numel() for k in head_keys)

    print(f"Client loaded: {client.client_id} ({client.domain}) | train={client.num_train_samples}, val={len(client.val_dataset)}, test={len(client.test_dataset)}")
    print(f"Total trainable tensors: {len(trainable_state)}")
    print(f"  LoRA tensors: {len(lora_keys)} ({lora_params:,} params)")
    print(f"  Head tensors: {len(head_keys)} ({head_params:,} params)")
    print("  Head key names:", head_keys)
    print("FederatedClient self-test PASSED!")


if __name__ == "__main__":
    _run_self_test()