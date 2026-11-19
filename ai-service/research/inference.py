import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Ensure ai-service root is on sys.path when run directly
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from research.reliability import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    assess_prediction_reliability,
)

MODELS_DIR = BASE_DIR / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

BASE_MODEL_NAME = "FacebookAI/roberta-base"
MAX_LENGTH = 256
ID2LABEL = {0: "Non-Stress", 1: "Stress"}
LABEL2ID = {"Non-Stress": 0, "Stress": 1}

VALID_CLIENTS = ["client_1", "client_2", "client_3", "client_4", "client_5"]


class StressInferenceEngine:
    """
    Loads one shared RoBERTa-base backbone and attaches all trained LoRA adapters
    (FedAvg Global, Centralized Baseline, and Ditto-LoRA Personalized Client Adapters)
    for fast, low-VRAM calibrated stress detection inference.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model: Optional[PeftModel] = None
        self.loaded_adapters: Dict[str, str] = {}
        self.temperatures: Dict[str, float] = {
            "federated_global": 1.0,
            "centralized_lora": 1.0,
            "personalized_ditto": 1.0,
        }
        self.is_loaded = False

    def _load_temperatures(self) -> None:
        rel_path = EXPERIMENTS_DIR / "reliability_analysis.json"
        if rel_path.exists():
            with open(rel_path, "r", encoding="utf-8") as f:
                rel_data = json.load(f)
            models_rel = rel_data.get("models", {})
            for key in ["federated_global", "centralized_lora", "personalized_ditto"]:
                if key in models_rel:
                    self.temperatures[key] = float(models_rel[key]["optimal_temperature"])

    def load(self) -> None:
        if self.is_loaded:
            return

        self._load_temperatures()

        global_dir = MODELS_DIR / "federated_global_lora"
        centralized_dir = MODELS_DIR / "centralized_lora"
        ditto_root = MODELS_DIR / "personalized_ditto_lora"

        if not global_dir.exists():
            raise FileNotFoundError(f"FedAvg global model not found at: {global_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(global_dir))
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        # Load FedAvg global adapter first as default adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            str(global_dir),
            adapter_name="federated_global",
            is_trainable=False,
        )
        self.loaded_adapters["federated_global"] = str(global_dir)

        # Attach centralized baseline adapter if available
        if centralized_dir.exists():
            self.model.load_adapter(str(centralized_dir), adapter_name="centralized_lora")
            self.loaded_adapters["centralized_lora"] = str(centralized_dir)

        # Attach all 5 Ditto-LoRA personalized client adapters if available
        for cid in VALID_CLIENTS:
            c_dir = ditto_root / cid
            if c_dir.exists():
                adapter_name = f"ditto_{cid}"
                self.model.load_adapter(str(c_dir), adapter_name=adapter_name)
                self.loaded_adapters[adapter_name] = str(c_dir)

        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

    def _resolve_adapter_and_temperature(
        self,
        model_type: str,
        client_id: Optional[str],
    ) -> tuple[str, float, str]:
        m_type = (model_type or "personalized_ditto").strip().lower()

        if m_type == "centralized_lora" and "centralized_lora" in self.loaded_adapters:
            return "centralized_lora", self.temperatures["centralized_lora"], "Centralized RoBERTa + LoRA"

        if m_type == "federated_global":
            return "federated_global", self.temperatures["federated_global"], "FedAvg Global RoBERTa + LoRA"

        if m_type == "personalized_ditto":
            cid = (client_id or "client_2").strip().lower()
            adapter_name = f"ditto_{cid}"
            if adapter_name in self.loaded_adapters:
                return (
                    adapter_name,
                    self.temperatures["personalized_ditto"],
                    f"Ditto-LoRA Personalized ({cid})",
                )
            # Fallback to federated_global if unknown client_id
            return "federated_global", self.temperatures["federated_global"], "FedAvg Global RoBERTa + LoRA"

        return "federated_global", self.temperatures["federated_global"], "FedAvg Global RoBERTa + LoRA"

    def predict(
        self,
        text: str,
        model_type: str = "personalized_ditto",
        client_id: Optional[str] = "client_2",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> dict:
        if not self.is_loaded:
            self.load()

        cleaned_text = " ".join(str(text).split()).strip()
        if not cleaned_text:
            raise ValueError("Input text cannot be empty.")

        adapter_name, temperature, model_label = self._resolve_adapter_and_temperature(
            model_type=model_type,
            client_id=client_id,
        )
        self.model.set_adapter(adapter_name)

        encoded = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float().cpu().numpy()[0]

        result = assess_prediction_reliability(
            logits=logits,
            temperature=temperature,
            confidence_threshold=confidence_threshold,
        )
        result["model_used"] = model_label
        result["adapter_name"] = adapter_name
        result["input_word_count"] = len(cleaned_text.split())
        return result


def _run_self_test() -> None:
    print("Loading StressInferenceEngine (multi-adapter RoBERTa + LoRA)...")
    engine = StressInferenceEngine()
    engine.load()
    print(f"Device: {engine.device}")
    print(f"Loaded adapters ({len(engine.loaded_adapters)}): {list(engine.loaded_adapters.keys())}")
    print(f"Calibration temperatures: {engine.temperatures}")

    samples = [
        (
            "Clear Stress",
            "I cannot sleep anymore, my chest feels tight every morning, and I am completely overwhelmed by panic and deadlines.",
        ),
        (
            "Clear Non-Stress",
            "Had a relaxing weekend reading a book in the park and catching up with an old friend over coffee.",
        ),
        (
            "Borderline / Ambiguous",
            "Work has been busy lately and I have a lot on my plate, though I think I might manage it next week.",
        ),
    ]

    print("\nRunning sample predictions (Ditto-LoRA client_2 vs. FedAvg Global):")
    for label, text in samples:
        res_ditto = engine.predict(text, model_type="personalized_ditto", client_id="client_2")
        res_global = engine.predict(text, model_type="federated_global")
        print(f"\n[{label}]")
        print(
            f"  Ditto (client_2): {res_ditto['prediction']:<10} | "
            f"Conf: {res_ditto['confidence']:.4f} | Entropy: {res_ditto['normalized_entropy']:.4f} | "
            f"Human Review: {res_ditto['needs_human_review']}"
        )
        print(
            f"  FedAvg Global:    {res_global['prediction']:<10} | "
            f"Conf: {res_global['confidence']:.4f} | Entropy: {res_global['normalized_entropy']:.4f} | "
            f"Human Review: {res_global['needs_human_review']}"
        )

    print("\nStressInferenceEngine self-test PASSED!")


if __name__ == "__main__":
    _run_self_test()