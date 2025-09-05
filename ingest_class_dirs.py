# ingest_class_dirs.py
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

from extract_and_build import extract_fields_from_text, build_text  # 直接重用你已有的函式

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="raw", help="含有多個類別子資料夾的根目錄")
    ap.add_argument("--out_csv", type=str, default="data/processed/records.csv")
    ap.add_argument("--out_jsonl", type=str, default="data/processed/records.jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for label_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = label_dir.name.strip()
        txts = sorted(label_dir.glob("*.txt"))
        for p in tqdm(txts, desc=f"[{label}]"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            fields = extract_fields_from_text(text)
            row = {
                "doc_id": p.stem,
                "label": label,
                **fields
            }
            row["built_text"] = build_text(row)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")

    jpath = Path(args.out_jsonl)
    with jpath.open("w", encoding="utf-8") as f:
        for _, r in out_df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"[OK] 匯入完成：{len(out_df)} 筆 → {out_path}")
    print(f"[OK] 同步輸出 JSONL：{jpath}")

if __name__ == "__main__":
    main()
