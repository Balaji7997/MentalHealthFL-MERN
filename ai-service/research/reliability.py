import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"

# Default operating policy for flagging uncertain predictions for human review
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def logits_to_probs(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature scaling to raw logits and convert to probabilities."""
    safe_temp = max(float(temperature), 1e-4)
    return softmax(logits / safe_temp, axis=-1)


def compute_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Compute Negative Log-Likelihood (cross-entropy loss) for binary classification."""
    probs = logits_to_probs(logits, temperature)
    true_probs = np.clip(probs[np.arange(len(labels)), labels.astype(int)], 1e-12, 1.0)
    return float(-np.mean(np.log(true_probs)))


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Fit optimal scalar temperature T > 0 strictly on validation logits by minimizing NLL."""
    result = minimize_scalar(
        lambda t: compute_nll(val_logits, val_labels, t),
        bounds=(0.25, 10.0),
        method="bounded",
    )
    return round(float(result.x), 4)


def compute_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    """Compute binary predictive entropy normalized to [0, 1] (0 = certain, 1 = max uncertainty)."""
    p = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)
    return entropy / np.log(2.0)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary Brier score: mean squared error between P(Stress) and true binary label."""
    prob_stress = probs[:, 1]
    return float(np.mean((prob_stress - labels.astype(float)) ** 2))


def compute_ece_and_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[dict]]:
    """Compute Expected Calibration Error (ECE) and per-bin reliability diagram data."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    correct = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_data = []
    n_total = len(labels)

    for i in range(n_bins):
        low = float(bin_edges[i])
        high = float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = int(np.sum(mask))
        if count > 0:
            avg_conf = float(np.mean(confidences[mask]))
            avg_acc = float(np.mean(correct[mask]))
            gap = abs(avg_acc - avg_conf)
            ece += (count / n_total) * gap
        else:
            avg_conf = 0.0
            avg_acc = 0.0
            gap = 0.0

        bins_data.append({
            "bin_index": i + 1,
            "range": f"{low:.1f}-{high:.1f}",
            "count": count,
            "avg_confidence": round(avg_conf, 4),
            "empirical_accuracy": round(avg_acc, 4),
            "calibration_gap": round(gap, 4),
        })

    return round(float(ece), 4), bins_data


def evaluate_threshold_tradeoff(
    probs: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float],
) -> List[dict]:
    """Evaluate coverage, human-review rate, automated accuracy, and error rate across thresholds."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    correct = predictions == labels
    n_total = len(labels)

    results = []
    for tau in thresholds:
        auto_mask = confidences >= tau
        review_mask = ~auto_mask

        n_auto = int(np.sum(auto_mask))
        n_review = int(np.sum(review_mask))
        coverage = n_auto / n_total
        review_rate = n_review / n_total

        if n_auto > 0:
            auto_acc = float(np.mean(correct[auto_mask]))
            auto_err = 1.0 - auto_acc
        else:
            auto_acc = 0.0
            auto_err = 0.0

        if n_review > 0:
            review_error_rate = float(np.mean(~correct[review_mask]))
        else:
            review_error_rate = 0.0

        results.append({
            "confidence_threshold": round(float(tau), 2),
            "automated_count": n_auto,
            "human_review_count": n_review,
            "coverage": round(coverage, 4),
            "human_review_rate": round(review_rate, 4),
            "automated_accuracy": round(auto_acc, 4),
            "automated_error_rate": round(auto_err, 4),
            "flagged_subset_error_rate": round(review_error_rate, 4),
        })

    return results


def assess_prediction_reliability(
    logits: np.ndarray,
    temperature: float = 1.0,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Reusable single-sample reliability assessment for FastAPI & Voice Assistant."""
    logits_1d = np.asarray(logits, dtype=np.float64).reshape(1, -1)
    probs = logits_to_probs(logits_1d, temperature=temperature)[0]

    prob_non_stress = float(probs[0])
    prob_stress = float(probs[1])
    pred_id = int(np.argmax(probs))
    pred_label = "Stress" if pred_id == 1 else "Non-Stress"

    confidence = float(np.max(probs))
    norm_entropy = float(compute_normalized_entropy(probs.reshape(1, -1))[0])
    reliability_score = float(1.0 - norm_entropy)

    needs_human_review = confidence < confidence_threshold
    if confidence >= 0.85:
        reliability_level = "High"
    elif confidence >= confidence_threshold:
        reliability_level = "Moderate"
    else:
        reliability_level = "Low (Uncertain - Human Review Required)"

    warning_message = (
        "Prediction uncertainty exceeds the reliability threshold. "
        "Flagged for human review; do not treat as a conclusive result."
        if needs_human_review
        else None
    )

    return {
        "prediction": pred_label,
        "prediction_id": pred_id,
        "probabilities": {
            "non_stress": round(prob_non_stress, 4),
            "stress": round(prob_stress, 4),
        },
        "confidence": round(confidence, 4),
        "normalized_entropy": round(norm_entropy, 4),
        "reliability_score": round(reliability_score, 4),
        "reliability_level": reliability_level,
        "calibration_temperature": round(float(temperature), 4),
        "confidence_threshold": round(float(confidence_threshold), 2),
        "needs_human_review": bool(needs_human_review),
        "warning": warning_message,
        "disclaimer": (
            "Research prototype for text-based stress indication only; "
            "not a medical or clinical diagnosis."
        ),
    }


def analyze_model_reliability(
    model_key: str,
    display_name: str,
    val_csv: Path,
    test_csv: Path,
    thresholds: List[float],
) -> dict:
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    val_logits = val_df[["logit_non_stress", "logit_stress"]].values
    val_labels = val_df["true_label"].values.astype(int)

    test_logits = test_df[["logit_non_stress", "logit_stress"]].values
    test_labels = test_df["true_label"].values.astype(int)

    # Uncalibrated probabilities (T = 1.0)
    raw_probs = logits_to_probs(test_logits, temperature=1.0)
    raw_nll = round(compute_nll(test_logits, test_labels, temperature=1.0), 4)
    raw_brier = round(compute_brier_score(raw_probs, test_labels), 4)
    raw_ece, raw_bins = compute_ece_and_bins(raw_probs, test_labels, n_bins=10)
    raw_entropy = compute_normalized_entropy(raw_probs)

    # Fit temperature strictly on validation logits (zero test leakage)
    opt_temp = fit_temperature(val_logits, val_labels)
    cal_probs = logits_to_probs(test_logits, temperature=opt_temp)
    cal_nll = round(compute_nll(test_logits, test_labels, temperature=opt_temp), 4)
    cal_brier = round(compute_brier_score(cal_probs, test_labels), 4)
    cal_ece, cal_bins = compute_ece_and_bins(cal_probs, test_labels, n_bins=10)
    cal_entropy = compute_normalized_entropy(cal_probs)

    preds = np.argmax(cal_probs, axis=-1)
    correct_mask = preds == test_labels

    mean_entropy_correct = round(float(np.mean(cal_entropy[correct_mask])), 4)
    mean_entropy_incorrect = round(float(np.mean(cal_entropy[~correct_mask])), 4)
    mean_conf_correct = round(float(np.mean(np.max(cal_probs[correct_mask], axis=-1))), 4)
    mean_conf_incorrect = round(float(np.mean(np.max(cal_probs[~correct_mask], axis=-1))), 4)

    # Per-client calibration summary if client_id column exists (personalized models)
    per_client_reliability = None
    if "client_id" in test_df.columns:
        per_client_reliability = {}
        for cid, group in test_df.groupby("client_id"):
            idx = group.index.values
            c_raw_probs = raw_probs[idx]
            c_cal_probs = cal_probs[idx]
            c_labels = test_labels[idx]
            c_raw_ece, _ = compute_ece_and_bins(c_raw_probs, c_labels, n_bins=10)
            c_cal_ece, _ = compute_ece_and_bins(c_cal_probs, c_labels, n_bins=10)
            c_confs = np.max(c_cal_probs, axis=-1)
            c_review_rate = float(np.mean(c_confs < DEFAULT_CONFIDENCE_THRESHOLD))
            per_client_reliability[cid] = {
                "domain": str(group["domain"].iloc[0]),
                "num_samples": int(len(idx)),
                "uncalibrated_ece": c_raw_ece,
                "calibrated_ece": c_cal_ece,
                "uncalibrated_brier": round(compute_brier_score(c_raw_probs, c_labels), 4),
                "calibrated_brier": round(compute_brier_score(c_cal_probs, c_labels), 4),
                "human_review_rate_at_default_threshold": round(c_review_rate, 4),
            }

    tradeoff_curve = evaluate_threshold_tradeoff(cal_probs, test_labels, thresholds)
    default_point = next(
        (pt for pt in tradeoff_curve if pt["confidence_threshold"] == DEFAULT_CONFIDENCE_THRESHOLD),
        tradeoff_curve[0],
    )

    return {
        "model_key": model_key,
        "display_name": display_name,
        "test_samples": int(len(test_labels)),
        "test_accuracy": round(float(np.mean(correct_mask)), 4),
        "calibration_method": "Temperature Scaling (fitted on validation split only)",
        "optimal_temperature": opt_temp,
        "uncalibrated": {
            "nll": raw_nll,
            "ece": raw_ece,
            "brier_score": raw_brier,
            "mean_normalized_entropy": round(float(np.mean(raw_entropy)), 4),
            "reliability_bins": raw_bins,
        },
        "calibrated": {
            "nll": cal_nll,
            "ece": cal_ece,
            "brier_score": cal_brier,
            "mean_normalized_entropy": round(float(np.mean(cal_entropy)), 4),
            "reliability_bins": cal_bins,
        },
        "calibration_improvement": {
            "ece_delta": round(cal_ece - raw_ece, 4),
            "brier_delta": round(cal_brier - raw_brier, 4),
            "nll_delta": round(cal_nll - raw_nll, 4),
        },
        "uncertainty_separation": {
            "mean_entropy_correct": mean_entropy_correct,
            "mean_entropy_incorrect": mean_entropy_incorrect,
            "mean_confidence_correct": mean_conf_correct,
            "mean_confidence_incorrect": mean_conf_incorrect,
        },
        "default_human_review_policy": default_point,
        "threshold_tradeoff_curve": tradeoff_curve,
        "per_client_reliability": per_client_reliability,
    }


def main():
    models_to_analyze = [
        (
            "centralized_lora",
            "Centralized RoBERTa + LoRA",
            EXPERIMENTS_DIR / "centralized_val_predictions.csv",
            EXPERIMENTS_DIR / "centralized_test_predictions.csv",
        ),
        (
            "federated_global",
            "FedAvg Global RoBERTa + LoRA",
            EXPERIMENTS_DIR / "federated_global_val_predictions.csv",
            EXPERIMENTS_DIR / "federated_global_test_predictions.csv",
        ),
        (
            "personalized_fedper",
            "FedPer-LoRA Personalized",
            EXPERIMENTS_DIR / "personalized_val_predictions.csv",
            EXPERIMENTS_DIR / "personalized_test_predictions.csv",
        ),
        (
            "personalized_ditto",
            "Ditto-LoRA Personalized",
            EXPERIMENTS_DIR / "personalized_ditto_val_predictions.csv",
            EXPERIMENTS_DIR / "personalized_ditto_test_predictions.csv",
        ),
    ]

    thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    print("=" * 84)
    print("PHASE 6: RELIABILITY, UNCERTAINTY ESTIMATION & TEMPERATURE SCALING CALIBRATION")
    print("=" * 84)

    all_results: Dict[str, dict] = {}

    for key, name, val_csv, test_csv in models_to_analyze:
        res = analyze_model_reliability(key, name, val_csv, test_csv, thresholds)
        all_results[key] = res

        print(f"\n--- {name} (Test Acc: {res['test_accuracy']:.4f}, Optimal T = {res['optimal_temperature']:.4f}) ---")
        print(
            f"  Uncalibrated | ECE: {res['uncalibrated']['ece']:.4f} | "
            f"Brier: {res['uncalibrated']['brier_score']:.4f} | NLL: {res['uncalibrated']['nll']:.4f}"
        )
        print(
            f"  Calibrated   | ECE: {res['calibrated']['ece']:.4f} ({res['calibration_improvement']['ece_delta']:+.4f}) | "
            f"Brier: {res['calibrated']['brier_score']:.4f} ({res['calibration_improvement']['brier_delta']:+.4f}) | "
            f"NLL: {res['calibrated']['nll']:.4f} ({res['calibration_improvement']['nll_delta']:+.4f})"
        )
        sep = res["uncertainty_separation"]
        print(
            f"  Uncertainty Separation | Correct Entropy: {sep['mean_entropy_correct']:.4f} (Conf {sep['mean_confidence_correct']:.4f}) "
            f"vs. Incorrect Entropy: {sep['mean_entropy_incorrect']:.4f} (Conf {sep['mean_confidence_incorrect']:.4f})"
        )

    ditto = all_results["personalized_ditto"]
    print("\n" + "=" * 84)
    print("HUMAN-REVIEW COVERAGE vs. ERROR TRADE-OFF (Calibrated Ditto-LoRA Personalized)")
    print("=" * 84)
    print(f"{'Threshold':>9} | {'Coverage':>8} | {'Flagged%':>8} | {'Auto Acc':>8} | {'Auto Err':>8} | {'Flagged Err':>11}")
    print("-" * 84)
    for pt in ditto["threshold_tradeoff_curve"]:
        print(
            f"{pt['confidence_threshold']:>9.2f} | "
            f"{pt['coverage']:>7.2%} | "
            f"{pt['human_review_rate']:>7.2%} | "
            f"{pt['automated_accuracy']:>7.2%} | "
            f"{pt['automated_error_rate']:>7.2%} | "
            f"{pt['flagged_subset_error_rate']:>10.2%}"
        )

    sample_check = assess_prediction_reliability(
        logits=np.array([0.4, 0.6]),
        temperature=ditto["optimal_temperature"],
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    print("\nLive Reliability Helper Sanity Check (borderline logits [0.4, 0.6]):")
    print(
        f"  pred={sample_check['prediction']} | conf={sample_check['confidence']} | "
        f"entropy={sample_check['normalized_entropy']} | human_review={sample_check['needs_human_review']}"
    )

    output_payload = {
        "phase": "Phase 6 - Reliability, Uncertainty Estimation & Calibration",
        "calibration_method": "Temperature Scaling fitted strictly on validation logits",
        "uncertainty_metric": "Normalized Predictive Entropy H(p)/ln(2) and Max Calibrated Confidence",
        "default_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "recommended_deployment_model": "personalized_ditto",
        "disclaimer": (
            "This reliability mechanism flags uncertain predictions for human review. "
            "It is a research prototype and does not provide medical or clinical reliability guarantees."
        ),
        "models": all_results,
    }

    out_path = EXPERIMENTS_DIR / "reliability_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved full reliability & calibration analysis to: {out_path}")


if __name__ == "__main__":
    main()