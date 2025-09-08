# mask_label_terms.py
import re
import argparse
import pandas as pd
from pathlib import Path

# 針對你的九類：盡量涵蓋常見變體/同義寫法
PATTERNS = {
    "保單查詢": [r"保單查詢"],
    "保單查詢＋註記": [r"查詢[及和並後]*註記", r"註記[及和並後]*查詢"],
    "保單註記": [r"保單註記", r"維持註記", r"延長註記", r"警示", r"限制註記?"],
    "公職查詢": [r"公職人員", r"政風", r"廉政", r"財產申報"],
    "扣押命令": [r"扣押", r"支付轉給命令", r"禁止給付", r"強制執行法", r"執行命令", r"執行處"],
    "撤銷令": [r"撤銷(令|命令)?", r"撤回", r"撤銷前令"],
    "收取＋撤銷": [r"(收取|代收|逕收).*(撤銷)|(撤銷).*(收取|代收|逕收)"],
    "收取令": [r"收取令", r"收取", r"代收", r"逕收", r"先行收取", r"先予收取"],
    "通知函": [r"通知函", r"轉知", r"知會", r"請查照", r"如說明辦理"],
}

MASK_RE = re.compile("|".join(f"(?:{p})" for pats in PATTERNS.values() for p in pats))

def mask_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return MASK_RE.sub("[MASK]", s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", default="data/processed/records.csv")
    ap.add_argument("--text_col", default="built_text")
    ap.add_argument("--output_csv", default="data/processed/records_masked.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"找不到欄位 {args.text_col}")
    df["built_text_masked"] = df[args.text_col].astype(str).map(mask_text)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[OK] 已輸出遮蔽版：{args.output_csv}")

if __name__ == "__main__":
    main()
