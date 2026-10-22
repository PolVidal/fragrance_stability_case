"""Convert the raw CSV to a compressed Parquet file and print a quick profile.

Usage (from the repository root):
    python3 scripts/convert_to_parquet.py path/to/fragrance_stability_dataset.csv

Output: data/raw/fragrance_stability.parquet (small enough to commit to a private repo).
Optional: --sample 0.2 also writes a batch-level sample (all rows of 20% of batch_ids).
"""
import argparse
import os
import sys

import pandas as pd

try:
    import pyarrow  # noqa: F401
except ImportError:
    sys.exit("pyarrow is missing. Run: pip3 install pandas pyarrow")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="path to the original CSV")
    ap.add_argument("--out", default="data/raw/fragrance_stability.parquet")
    ap.add_argument("--sample", type=float, default=None,
                    help="optional fraction of batch_ids to keep in an extra sample file")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"File not found: {args.csv}")

    print(f"Reading {args.csv} ...")
    df = pd.read_csv(args.csv, low_memory=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, compression="zstd", index=False)

    csv_mb = os.path.getsize(args.csv) / 1e6
    pq_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out}: {pq_mb:.1f} MB (CSV was {csv_mb:.1f} MB)")
    if pq_mb > 95:
        print("WARNING: GitHub rejects files over 100 MB. Use --sample or Git LFS.")

    # Quick profile of the converted table
    print("\nShape:", df.shape)
    print("\nColumns and types:\n", df.dtypes.to_string())
    print("\nMissing values per column:\n", df.isna().sum().to_string())
    if "batch_id" in df.columns:
        n_batches = df["batch_id"].nunique()
        print(f"\nUnique batch_id: {n_batches} (rows per batch: {len(df) / n_batches:.2f})")
    for col in ("plant_country", "production_line_id"):
        if col in df.columns:
            print(f"\n{col} counts:\n", df[col].value_counts(dropna=False).to_string())
    print("\nNumeric summary:\n", df.describe().T.to_string())

    if args.sample:
        ids = df["batch_id"].drop_duplicates().sample(frac=args.sample, random_state=42)
        sample_path = args.out.replace(".parquet", f"_sample{int(args.sample * 100)}.parquet")
        df[df["batch_id"].isin(ids)].to_parquet(sample_path, compression="zstd", index=False)
        print(f"\nWrote sample {sample_path}: {os.path.getsize(sample_path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
