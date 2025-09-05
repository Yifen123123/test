# split_dataset.py
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="data/processed/records.csv")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--train_size", type=float, default=0.7)
    ap.add_argument("--valid_size", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=[args.label_col])

    # 先切 train / temp
    train_df, temp_df = train_test_split(
        df, test_size=(1 - args.train_size), 
        stratify=df[args.label_col], random_state=42
    )
    # 再切 valid / test
    valid_frac = args.valid_size / (1 - args.train_size)
    valid_df, test_df = train_test_split(
        temp_df, test_size=(1 - valid_frac),
        stratify=temp_df[args.label_col], random_state=42
    )

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir/"train.csv", index=False, encoding="utf-8")
    valid_df.to_csv(out_dir/"valid.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir/"test.csv", index=False, encoding="utf-8")

    print(f"[OK] 切分完成：")
    print(f"- Train: {len(train_df)}")
    print(f"- Valid: {len(valid_df)}")
    print(f"- Test : {len(test_df)}")

if __name__ == "__main__":
    main()
