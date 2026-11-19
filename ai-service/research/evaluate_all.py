import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EXPERIMENTS_DIR = BASE_DIR / "experiments"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    dataset_summary = load_json(DATA_DIR / "processed" / "dataset_summary.json")
    partition_summary = load_json(DATA_DIR / "clients" / "partition_summary.json")
    centralized = load_json(EXPERIMENTS_DIR / "centralized_baseline_metrics.json")
    fedavg = load_json(EXPERIMENTS_DIR / "federated_fedavg_metrics.json")
    fedper = load_json(EXPERIMENTS_DIR / "personalized_fedper_metrics.json")
    ditto = load_json(EXPERIMENTS_DIR / "personalized_ditto_metrics.json")
    reliability = load_json(EXPERIMENTS_DIR / "reliability_analysis.json")

    rel_models = reliability["models"]

    # 1. Unified 4-model comparison table
    model_comparison = [
        {
            "model_key": "centralized_lora",
            "name": "Centralized RoBERTa + LoRA",
            "paradigm": "Centralized (Upper Bound)",
            "data_centralized": True,
            "upload_mb_per_client_round": None,
            "accuracy": centralized["test_metrics"]["accuracy"],
            "precision": centralized["test_metrics"]["precision"],
            "recall": centralized["test_metrics"]["recall"],
            "f1": centralized["test_metrics"]["f1"],
            "confusion_matrix": centralized["test_metrics"]["confusion_matrix"],
            "optimal_temperature": rel_models["centralized_lora"]["optimal_temperature"],
            "uncalibrated_ece": rel_models["centralized_lora"]["uncalibrated"]["ece"],
            "calibrated_ece": rel_models["centralized_lora"]["calibrated"]["ece"],
            "uncalibrated_brier": rel_models["centralized_lora"]["uncalibrated"]["brier_score"],
            "calibrated_brier": rel_models["centralized_lora"]["calibrated"]["brier_score"],
            "calibrated_nll": rel_models["centralized_lora"]["calibrated"]["nll"],
            "default_policy_coverage": rel_models["centralized_lora"]["default_human_review_policy"]["coverage"],
            "default_policy_auto_acc": rel_models["centralized_lora"]["default_human_review_policy"]["automated_accuracy"],
        },
        {
            "model_key": "federated_global",
            "name": "FedAvg Global RoBERTa + LoRA",
            "paradigm": "Federated Learning (Global)",
            "data_centralized": False,
            "upload_mb_per_client_round": fedavg["communication_efficiency"]["upload_size_per_client_mb"],
            "accuracy": fedavg["final_global_test_metrics"]["accuracy"],
            "precision": fedavg["final_global_test_metrics"]["precision"],
            "recall": fedavg["final_global_test_metrics"]["recall"],
            "f1": fedavg["final_global_test_metrics"]["f1"],
            "confusion_matrix": fedavg["final_global_test_metrics"]["confusion_matrix"],
            "optimal_temperature": rel_models["federated_global"]["optimal_temperature"],
            "uncalibrated_ece": rel_models["federated_global"]["uncalibrated"]["ece"],
            "calibrated_ece": rel_models["federated_global"]["calibrated"]["ece"],
            "uncalibrated_brier": rel_models["federated_global"]["uncalibrated"]["brier_score"],
            "calibrated_brier": rel_models["federated_global"]["calibrated"]["brier_score"],
            "calibrated_nll": rel_models["federated_global"]["calibrated"]["nll"],
            "default_policy_coverage": rel_models["federated_global"]["default_human_review_policy"]["coverage"],
            "default_policy_auto_acc": rel_models["federated_global"]["default_human_review_policy"]["automated_accuracy"],
        },
        {
            "model_key": "personalized_fedper",
            "name": "FedPer-LoRA (Private Heads)",
            "paradigm": "Personalized FL (Architectural)",
            "data_centralized": False,
            "upload_mb_per_client_round": fedper["parameter_split"]["shared_upload_mb_per_client_round"],
            "accuracy": fedper["overall_personalized_test_metrics"]["accuracy"],
            "precision": fedper["overall_personalized_test_metrics"]["precision"],
            "recall": fedper["overall_personalized_test_metrics"]["recall"],
            "f1": fedper["overall_personalized_test_metrics"]["f1"],
            "confusion_matrix": fedper["overall_personalized_test_metrics"]["confusion_matrix"],
            "optimal_temperature": rel_models["personalized_fedper"]["optimal_temperature"],
            "uncalibrated_ece": rel_models["personalized_fedper"]["uncalibrated"]["ece"],
            "calibrated_ece": rel_models["personalized_fedper"]["calibrated"]["ece"],
            "uncalibrated_brier": rel_models["personalized_fedper"]["uncalibrated"]["brier_score"],
            "calibrated_brier": rel_models["personalized_fedper"]["calibrated"]["brier_score"],
            "calibrated_nll": rel_models["personalized_fedper"]["calibrated"]["nll"],
            "default_policy_coverage": rel_models["personalized_fedper"]["default_human_review_policy"]["coverage"],
            "default_policy_auto_acc": rel_models["personalized_fedper"]["default_human_review_policy"]["automated_accuracy"],
        },
        {
            "model_key": "personalized_ditto",
            "name": "Ditto-LoRA (Proximal Personalization)",
            "paradigm": "Personalized FL (Regularized)",
            "data_centralized": False,
            "upload_mb_per_client_round": fedavg["communication_efficiency"]["upload_size_per_client_mb"],
            "accuracy": ditto["overall_personalized_test_metrics"]["accuracy"],
            "precision": ditto["overall_personalized_test_metrics"]["precision"],
            "recall": ditto["overall_personalized_test_metrics"]["recall"],
            "f1": ditto["overall_personalized_test_metrics"]["f1"],
            "confusion_matrix": ditto["overall_personalized_test_metrics"]["confusion_matrix"],
            "optimal_temperature": rel_models["personalized_ditto"]["optimal_temperature"],
            "uncalibrated_ece": rel_models["personalized_ditto"]["uncalibrated"]["ece"],
            "calibrated_ece": rel_models["personalized_ditto"]["calibrated"]["ece"],
            "uncalibrated_brier": rel_models["personalized_ditto"]["uncalibrated"]["brier_score"],
            "calibrated_brier": rel_models["personalized_ditto"]["calibrated"]["brier_score"],
            "calibrated_nll": rel_models["personalized_ditto"]["calibrated"]["nll"],
            "default_policy_coverage": rel_models["personalized_ditto"]["default_human_review_policy"]["coverage"],
            "default_policy_auto_acc": rel_models["personalized_ditto"]["default_human_review_policy"]["automated_accuracy"],
        },
    ]

    # 2. Client-wise comparison across FedAvg Global, FedPer-LoRA, and Ditto-LoRA
    client_comparison = []
    for c_meta in partition_summary["clients"]:
        cid = c_meta["client_id"]
        g_m = fedavg["per_client_test_metrics"][cid]
        fp_m = fedper["per_client_test_metrics"][cid]
        dt_m = ditto["per_client_test_metrics"][cid]
        rel_c = rel_models["personalized_ditto"]["per_client_reliability"][cid]

        client_comparison.append({
            "client_id": cid,
            "domain": c_meta["domain"],
            "display_name": c_meta["display_name"],
            "subreddits": c_meta["subreddits"],
            "train_samples": c_meta["train"]["rows"],
            "train_stress_ratio": c_meta["train"]["stress_ratio"],
            "test_samples": g_m["num_samples"],
            "fedavg_accuracy": g_m["accuracy"],
            "fedavg_f1": g_m["f1"],
            "fedper_accuracy": fp_m["accuracy"],
            "fedper_f1": fp_m["f1"],
            "ditto_accuracy": dt_m["accuracy"],
            "ditto_f1": dt_m["f1"],
            "ditto_vs_fedavg_acc_delta": round(dt_m["accuracy"] - g_m["accuracy"], 4),
            "ditto_vs_fedavg_f1_delta": round(dt_m["f1"] - g_m["f1"], 4),
            "ditto_calibrated_ece": rel_c["calibrated_ece"],
            "ditto_calibrated_brier": rel_c["calibrated_brier"],
            "ditto_human_review_rate": rel_c["human_review_rate_at_default_threshold"],
        })

    evaluation_summary = {
        "project_title": "Reliability-Aware Personalized Federated Learning for Mental Health Stress Detection",
        "dataset": {
            "name": dataset_summary["dataset_name"],
            "source": dataset_summary["source"],
            "citation": dataset_summary["citation"],
            "train_rows": dataset_summary["splits"]["train"]["rows"],
            "validation_rows": dataset_summary["splits"]["validation"]["rows"],
            "test_rows": dataset_summary["splits"]["test"]["rows"],
            "num_clients": partition_summary["num_clients"],
            "non_iid_stress_spread": partition_summary["train_stress_ratio_range"],
        },
        "base_architecture": {
            "base_model": centralized["base_model"],
            "total_parameters": centralized["parameters"]["total_parameters"],
            "trainable_parameters": centralized["parameters"]["trainable_parameters"],
            "trainable_percentage": centralized["parameters"]["trainable_percentage"],
            "lora_r": centralized["hyperparameters"]["lora_r"],
            "lora_alpha": centralized["hyperparameters"]["lora_alpha"],
            "lora_target_modules": centralized["hyperparameters"]["lora_target_modules"],
        },
        "model_comparison": model_comparison,
        "client_comparison": client_comparison,
        "human_review_policy": {
            "recommended_model": "personalized_ditto",
            "default_confidence_threshold": reliability["default_confidence_threshold"],
            "optimal_temperature": rel_models["personalized_ditto"]["optimal_temperature"],
            "default_operating_point": rel_models["personalized_ditto"]["default_human_review_policy"],
            "tradeoff_curve": rel_models["personalized_ditto"]["threshold_tradeoff_curve"],
        },
        "privacy_and_scope_notes": [
            "Federated learning avoids pooling raw client text centrally, but weight updates alone do not guarantee absolute privacy without additional defenses (e.g., differential privacy or secure aggregation).",
            "This system is a research prototype for text-based stress indication with uncertainty flagging; it is not a clinical or medical diagnostic tool.",
        ],
    }

    out_path = EXPERIMENTS_DIR / "evaluation_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)

    print("=" * 94)
    print("PHASE 7: CONSOLIDATED MODEL EVALUATION SUMMARY (DREADDIT TEST SET, n=715)")
    print("=" * 94)
    print(f"{'Model':<38} | {'Acc':>6} | {'Prec':>6} | {'Rec':>6} | {'F1':>6} | {'ECE(Cal)':>8} | {'Brier':>6}")
    print("-" * 94)
    for m in model_comparison:
        print(
            f"{m['name']:<38} | {m['accuracy']:>6.4f} | {m['precision']:>6.4f} | "
            f"{m['recall']:>6.4f} | {m['f1']:>6.4f} | {m['calibrated_ece']:>8.4f} | {m['calibrated_brier']:>6.4f}"
        )

    print("\nCLIENT-WISE COMPARISON (FedAvg Global vs. FedPer-LoRA vs. Ditto-LoRA):")
    print(f"{'Client':<10} | {'Domain':<9} | {'FedAvg F1':>9} | {'FedPer F1':>9} | {'Ditto F1':>8} | {'Delta':>7} | {'Review%':>7}")
    print("-" * 76)
    for c in client_comparison:
        print(
            f"{c['client_id']:<10} | {c['domain']:<9} | {c['fedavg_f1']:>9.4f} | "
            f"{c['fedper_f1']:>9.4f} | {c['ditto_f1']:>8.4f} | {c['ditto_vs_fedavg_f1_delta']:>+7.4f} | "
            f"{c['ditto_human_review_rate']:>6.1%}"
        )

    print(f"\nSaved master evaluation summary to: {out_path}")


if __name__ == "__main__":
    main()