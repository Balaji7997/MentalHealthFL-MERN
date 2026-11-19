from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from research.inference import StressInferenceEngine
from research.reliability import DEFAULT_CONFIDENCE_THRESHOLD

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

engine = StressInferenceEngine()


def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Required experiment artifact not found: {path.name}",
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load RoBERTa-base + all 7 trained LoRA adapters once on server startup
    engine.load()
    yield


app = FastAPI(
    title="MentalHealthFL AI Service",
    description="Reliability-Aware Personalized Federated Learning for Mental Health Stress Detection",
    version="2.0.0",
    lifespan=lifespan,
)

# Allow communication with React frontend (5173) and Express backend (5000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5000",
        "http://127.0.0.1:5000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000, description="Text to analyze for stress")
    model_type: str = Field(
        default="personalized_ditto",
        description="Model variant: 'personalized_ditto', 'federated_global', or 'centralized_lora'",
    )
    client_id: Optional[str] = Field(
        default="client_2",
        description="Client ID ('client_1' to 'client_5') used when model_type='personalized_ditto'",
    )
    confidence_threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        ge=0.50,
        le=0.99,
        description="Minimum calibrated confidence below which human review is triggered",
    )
    source: str = Field(default="text", description="Input channel: 'text' or 'voice'")
    language: str = Field(default="en", description="UI/voice language code: 'en' or 'te'")


@app.get("/")
def home():
    return {
        "message": "MentalHealthFL AI Service is running",
        "project": "Reliability-Aware Personalized Federated Learning for Mental Health Stress Detection",
        "status": "success",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Python AI Service",
        "device": str(engine.device),
        "model_status": "loaded" if engine.is_loaded else "not_loaded",
        "loaded_adapters": list(engine.loaded_adapters.keys()),
        "calibration_temperatures": engine.temperatures,
        "training_status": "completed",
    }


@app.get("/api/research")
def research_info():
    eval_summary = load_json_file(EXPERIMENTS_DIR / "evaluation_summary.json")
    return {
        "project": "MentalHealthFL",
        "title": eval_summary["project_title"],
        "dataset": "Dreaddit",
        "base_model": "FacebookAI/roberta-base",
        "fine_tuning": "LoRA (r=8, alpha=16, 0.8231% trainable)",
        "federated_learning": "FedAvg (5 Non-IID Domain Clients, 5 Rounds)",
        "personalization": "Ditto-LoRA (Proximal Personalization) & FedPer-LoRA",
        "reliability_assessment": "Temperature Scaling + Normalized Entropy + Human-Review Flagging",
        "clients": eval_summary["dataset"]["num_clients"],
        "default_confidence_threshold": eval_summary["human_review_policy"]["default_confidence_threshold"],
    }


@app.get("/api/overview")
def get_overview():
    eval_summary = load_json_file(EXPERIMENTS_DIR / "evaluation_summary.json")
    fedavg = load_json_file(EXPERIMENTS_DIR / "federated_fedavg_metrics.json")
    return {
        "project_title": eval_summary["project_title"],
        "dataset": eval_summary["dataset"],
        "base_architecture": eval_summary["base_architecture"],
        "model_comparison": eval_summary["model_comparison"],
        "human_review_policy": eval_summary["human_review_policy"],
        "communication_efficiency": fedavg["communication_efficiency"],
        "privacy_and_scope_notes": eval_summary["privacy_and_scope_notes"],
    }


@app.get("/api/clients")
def get_clients():
    partition = load_json_file(DATA_DIR / "clients" / "partition_summary.json")
    eval_summary = load_json_file(EXPERIMENTS_DIR / "evaluation_summary.json")
    return {
        "partition_strategy": partition["partition_strategy"],
        "num_clients": partition["num_clients"],
        "train_stress_ratio_range": partition["train_stress_ratio_range"],
        "clients_partition": partition["clients"],
        "client_comparison": eval_summary["client_comparison"],
    }


@app.get("/api/aggregation")
def get_aggregation():
    fedavg = load_json_file(EXPERIMENTS_DIR / "federated_fedavg_metrics.json")
    return {
        "aggregation_algorithm": fedavg["aggregation_algorithm"],
        "hyperparameters": fedavg["hyperparameters"],
        "communication_efficiency": fedavg["communication_efficiency"],
        "best_round": fedavg["best_round"],
        "rounds_history": fedavg["rounds_history"],
        "final_global_val_metrics": fedavg["final_global_val_metrics"],
        "final_global_test_metrics": fedavg["final_global_test_metrics"],
    }


@app.get("/api/personalization")
def get_personalization():
    fedper = load_json_file(EXPERIMENTS_DIR / "personalized_fedper_metrics.json")
    ditto = load_json_file(EXPERIMENTS_DIR / "personalized_ditto_metrics.json")
    eval_summary = load_json_file(EXPERIMENTS_DIR / "evaluation_summary.json")
    return {
        "recommended_method": "Ditto-LoRA",
        "ditto_lora": ditto,
        "fedper_lora": fedper,
        "client_comparison": eval_summary["client_comparison"],
    }


@app.get("/api/reliability")
def get_reliability():
    return load_json_file(EXPERIMENTS_DIR / "reliability_analysis.json")


@app.get("/api/evaluation")
def get_evaluation():
    eval_summary = load_json_file(EXPERIMENTS_DIR / "evaluation_summary.json")
    centralized = load_json_file(EXPERIMENTS_DIR / "centralized_baseline_metrics.json")
    return {
        "summary": eval_summary,
        "centralized_epoch_history": centralized["epoch_history"],
        "centralized_hyperparameters": centralized["hyperparameters"],
    }


@app.post("/api/predict")
def predict_stress(req: PredictRequest):
    cleaned = " ".join(req.text.split()).strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        result = engine.predict(
            text=cleaned,
            model_type=req.model_type,
            client_id=req.client_id,
            confidence_threshold=req.confidence_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

    result["input_channel"] = req.source
    result["language"] = req.language
    return result