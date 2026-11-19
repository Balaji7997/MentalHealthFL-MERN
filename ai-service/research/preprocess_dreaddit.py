import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Exact project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

# Official Dreaddit 5-domain mapping (Turcan & McKeown, LOUHI 2019)
SUBREDDIT_TO_DOMAIN = {
    "domesticviolence": "abuse",
    "survivorsofabuse": "abuse",
    "anxiety": "anxiety",
    "stress": "anxiety",
    "assistance": "financial",
    "homeless": "financial",
    "almosthomeless": "financial",
    "food_pantry": "financial",
    "ptsd": "ptsd",
    "relationships": "social",
}

ID2LABEL = {0: "Non-Stress", 1: "Stress"}


def clean_text(text: str) -> str:
    """Gentle text cleaning suitable for RoBERTa (preserves case and punctuation)."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove null bytes and non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Normalize all whitespace (newlines, tabs, multiple spaces) to a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_dataframe(df: pd.DataFrame, split_name: str, drop_duplicates: bool = False) -> tuple[pd.DataFrame, dict]:
    """Validate, clean, and select required research columns."""
    initial_rows = len(df)
    required_cols = ["post_id", "subreddit", "text", "label", "confidence"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {split_name}: {missing_cols}")

    out = df[required_cols].copy()
    out["text"] = out["text"].apply(clean_text)

    # Remove empty or invalid texts
    empty_mask = out["text"].str.len() == 0
    removed_empty = int(empty_mask.sum())
    out = out[~empty_mask].copy()

    # Keep only valid binary labels 0 and 1
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    invalid_label_mask = ~out["label"].isin([0, 1])
    removed_invalid_labels = int(invalid_label_mask.sum())
    out = out[~invalid_label_mask].copy()
    out["label"] = out["label"].astype(int)

    # Optionally remove exact duplicate texts inside training pool
    removed_duplicates = 0
    if drop_duplicates:
        before_dup = len(out)
        out = out.drop_duplicates(subset=["text"], keep="first").copy()
        removed_duplicates = before_dup - len(out)

    # Map labels and official Dreaddit domains
    out["label_name"] = out["label"].map(ID2LABEL)
    out["subreddit"] = out["subreddit"].astype(str).str.strip().str.lower()
    out["domain"] = out["subreddit"].map(SUBREDDIT_TO_DOMAIN).fillna("other")
    out["word_count"] = out["text"].apply(lambda s: len(s.split()))

    out = out[
        ["post_id", "subreddit", "domain", "text", "label", "label_name", "confidence", "word_count"]
    ].reset_index(drop=True)

    cleaning_log = {
        "initial_rows": initial_rows,
        "removed_empty_text": removed_empty,
        "removed_invalid_labels": removed_invalid_labels,
        "removed_exact_duplicates": removed_duplicates,
        "final_rows": len(out),
    }
    return out, cleaning_log


def split_stats(df: pd.DataFrame) -> dict:
    """Compute exact real statistics for a split."""
    label_counts = df["label"].value_counts().sort_index().to_dict()
    domain_counts = df["domain"].value_counts().sort_index().to_dict()
    subreddit_counts = df["subreddit"].value_counts().sort_index().to_dict()
    return {
        "rows": int(len(df)),
        "unique_posts": int(df["post_id"].nunique()),
        "non_stress_count": int(label_counts.get(0, 0)),
        "stress_count": int(label_counts.get(1, 0)),
        "stress_ratio": round(float(df["label"].mean()), 4),
        "avg_word_count": round(float(df["word_count"].mean()), 2),
        "min_word_count": int(df["word_count"].min()),
        "max_word_count": int(df["word_count"].max()),
        "domain_counts": {k: int(v) for k, v in domain_counts.items()},
        "subreddit_counts": {k: int(v) for k, v in subreddit_counts.items()},
    }


def main():
    raw_train_path = RAW_DIR / "dreaddit_train.csv"
    raw_test_path = RAW_DIR / "dreaddit_test.csv"

    print("Loading raw CSV files...")
    raw_train = pd.read_csv(raw_train_path)
    raw_test = pd.read_csv(raw_test_path)

    clean_train_pool, train_clean_log = prepare_dataframe(raw_train, "train", drop_duplicates=True)
    clean_test, test_clean_log = prepare_dataframe(raw_test, "test", drop_duplicates=False)

    # Check if any test texts appear in train pool (report only, keep official test intact)
    overlap_texts = len(set(clean_train_pool["text"]).intersection(set(clean_test["text"])))

    # Leak-free stratified group split by post_id (~80% train / ~20% validation)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(
        sgkf.split(clean_train_pool, clean_train_pool["label"], groups=clean_train_pool["post_id"])
    )

    train_df = clean_train_pool.iloc[train_idx].reset_index(drop=True)
    val_df = clean_train_pool.iloc[val_idx].reset_index(drop=True)

    # Verify zero post_id leakage between train and validation
    post_overlap = len(set(train_df["post_id"]).intersection(set(val_df["post_id"])))
    assert post_overlap == 0, f"Post ID leakage detected: {post_overlap}"

    # Save processed CSVs
    train_out = PROCESSED_DIR / "train.csv"
    val_out = PROCESSED_DIR / "validation.csv"
    test_out = PROCESSED_DIR / "test.csv"

    train_df.to_csv(train_out, index=False, encoding="utf-8")
    val_df.to_csv(val_out, index=False, encoding="utf-8")
    clean_test.to_csv(test_out, index=False, encoding="utf-8")

    # Build verified summary JSON
    summary = {
        "dataset_name": "Dreaddit",
        "source": "Hugging Face Hub (andreagasparini/dreaddit)",
        "citation": (
            "Turcan, E., & McKeown, K. (2019). Dreaddit: A Reddit Dataset for Stress "
            "Analysis in Social Media. Proceedings of the Tenth International Workshop "
            "on Health Text Mining and Information Analysis (LOUHI 2019), ACL."
        ),
        "task": "Binary Stress Detection",
        "label_mapping": {"0": "Non-Stress", "1": "Stress"},
        "random_seed": SEED,
        "split_method": "StratifiedGroupKFold (5-fold, grouped by post_id) on official train; official test preserved untouched",
        "train_val_post_id_overlap": post_overlap,
        "train_pool_test_exact_text_overlap": overlap_texts,
        "cleaning_log": {
            "raw_train_pool": train_clean_log,
            "raw_test": test_clean_log,
        },
        "splits": {
            "train": split_stats(train_df),
            "validation": split_stats(val_df),
            "test": split_stats(clean_test),
        },
    }

    summary_path = PROCESSED_DIR / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DREADDIT PREPROCESSING COMPLETE")
    print("=" * 60)
    print("Cleaning log (train pool):", train_clean_log)
    print("Cleaning log (test):      ", test_clean_log)
    print(f"Train/Val post_id overlap: {post_overlap} (leak-free verified)")
    print(f"Train/Test exact text overlap: {overlap_texts}")

    for name, stats in summary["splits"].items():
        print(f"\n--- {name.upper()} SPLIT ---")
        print(f"  Rows:         {stats['rows']} (unique posts: {stats['unique_posts']})")
        print(f"  Non-Stress 0: {stats['non_stress_count']}")
        print(f"  Stress 1:     {stats['stress_count']} (stress ratio: {stats['stress_ratio']})")
        print(f"  Words (avg):  {stats['avg_word_count']} (min={stats['min_word_count']}, max={stats['max_word_count']})")
        print(f"  Domains:      {stats['domain_counts']}")

    print("\nSaved files:")
    print(f"  - {train_out}")
    print(f"  - {val_out}")
    print(f"  - {test_out}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()