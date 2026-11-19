from pathlib import Path
from datasets import load_dataset
import pandas as pd

# Exact project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "andreagasparini/dreaddit"

print(f"Downloading Dreaddit dataset from Hugging Face: {DATASET_ID} ...")
dataset = load_dataset(DATASET_ID)

print("\nAvailable splits:", list(dataset.keys()))

for split_name in dataset.keys():
    df = dataset[split_name].to_pandas()
    out_path = RAW_DIR / f"dreaddit_{split_name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"SPLIT: {split_name}")
    print(f"Saved to: {out_path}")
    print(f"Shape (rows, columns): {df.shape}")
    print(f"First 15 columns: {list(df.columns[:15])}")

    # Check key columns
    for col in ["text", "label", "subreddit", "confidence", "post_id"]:
        if col in df.columns:
            missing = int(df[col].isna().sum())
            print(f"  Column '{col}': present | missing values = {missing}")
        else:
            print(f"  Column '{col}': NOT FOUND")

    if "label" in df.columns:
        print("\nLabel value counts:")
        print(df["label"].value_counts(dropna=False).sort_index().to_string())

    if "subreddit" in df.columns:
        print("\nSubreddit value counts:")
        print(df["subreddit"].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("Sample rows from train split (text shortened to 120 chars):")
train_df = pd.read_csv(RAW_DIR / "dreaddit_train.csv")
for i in range(min(3, len(train_df))):
    row = train_df.iloc[i]
    snippet = str(row["text"]).replace("\n", " ")[:120]
    safe_snippet = snippet.encode("ascii", errors="replace").decode("ascii")
    print(
        f"  Row {i} | label={row['label']} | subreddit={row.get('subreddit', 'N/A')} "
        f"| confidence={row.get('confidence', 'N/A')} | text='{safe_snippet}...'"
    )

print("\nDownload and raw inspection complete.")