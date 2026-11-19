import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLIENTS_DIR = BASE_DIR / "data" / "clients"
CLIENTS_DIR.mkdir(parents=True, exist_ok=True)

# 5 natural domain-based Non-IID federated clients (official Dreaddit domains)
CLIENT_DOMAINS = [
    ("client_1", "abuse", "Abuse & Domestic Violence Community"),
    ("client_2", "anxiety", "Anxiety & General Stress Community"),
    ("client_3", "financial", "Financial & Housing Distress Community"),
    ("client_4", "ptsd", "PTSD & Trauma Support Community"),
    ("client_5", "social", "Social & Relationships Community"),
]


def split_summary(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().sort_index().to_dict()
    return {
        "rows": int(len(df)),
        "non_stress": int(counts.get(0, 0)),
        "stress": int(counts.get(1, 0)),
        "stress_ratio": round(float(df["label"].mean()), 4) if len(df) > 0 else 0.0,
    }


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DIR / "validation.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    print("=" * 72)
    print("PHASE 4.1: NON-IID FEDERATED CLIENT PARTITIONING (5 DREADDIT DOMAINS)")
    print("=" * 72)

    clients_metadata = []
    total_train_assigned = 0
    total_val_assigned = 0
    total_test_assigned = 0

    for client_id, domain, display_name in CLIENT_DOMAINS:
        c_dir = CLIENTS_DIR / client_id
        c_dir.mkdir(parents=True, exist_ok=True)

        c_train = train_df[train_df["domain"] == domain].reset_index(drop=True)
        c_val = val_df[val_df["domain"] == domain].reset_index(drop=True)
        c_test = test_df[test_df["domain"] == domain].reset_index(drop=True)

        c_train.to_csv(c_dir / "train.csv", index=False, encoding="utf-8")
        c_val.to_csv(c_dir / "validation.csv", index=False, encoding="utf-8")
        c_test.to_csv(c_dir / "test.csv", index=False, encoding="utf-8")

        total_train_assigned += len(c_train)
        total_val_assigned += len(c_val)
        total_test_assigned += len(c_test)

        subreddits = sorted(c_train["subreddit"].unique().tolist())
        train_stats = split_summary(c_train)
        val_stats = split_summary(c_val)
        test_stats = split_summary(c_test)

        client_info = {
            "client_id": client_id,
            "domain": domain,
            "display_name": display_name,
            "subreddits": subreddits,
            "train": train_stats,
            "validation": val_stats,
            "test": test_stats,
        }
        clients_metadata.append(client_info)

        print(
            f"{client_id} | domain={domain:<9} | "
            f"train={train_stats['rows']:>3} (stress={train_stats['stress_ratio']:.2%}) | "
            f"val={val_stats['rows']:>3} | test={test_stats['rows']:>3} | "
            f"subreddits={subreddits}"
        )

    # Verify zero samples dropped or duplicated across clients
    assert total_train_assigned == len(train_df), "Train partition count mismatch!"
    assert total_val_assigned == len(val_df), "Validation partition count mismatch!"
    assert total_test_assigned == len(test_df), "Test partition count mismatch!"

    stress_ratios = [c["train"]["stress_ratio"] for c in clients_metadata]
    summary = {
        "partition_strategy": "Natural Domain-Based Non-IID (Feature/Topic + Label-Prevalence Skew)",
        "num_clients": len(clients_metadata),
        "total_train_samples": total_train_assigned,
        "total_validation_samples": total_val_assigned,
        "total_test_samples": total_test_assigned,
        "train_stress_ratio_range": {
            "min": min(stress_ratios),
            "max": max(stress_ratios),
            "spread": round(max(stress_ratios) - min(stress_ratios), 4),
        },
        "clients": clients_metadata,
    }

    summary_path = CLIENTS_DIR / "partition_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print(
        f"Verified 100% coverage -> Train: {total_train_assigned}, "
        f"Val: {total_val_assigned}, Test: {total_test_assigned}"
    )
    print(
        f"Non-IID Stress Ratio Spread across clients: "
        f"{summary['train_stress_ratio_range']['min']:.2%} to "
        f"{summary['train_stress_ratio_range']['max']:.2%} "
        f"(spread = {summary['train_stress_ratio_range']['spread']:.2%})"
    )
    print(f"Partition summary saved to: {summary_path}")


if __name__ == "__main__":
    main()