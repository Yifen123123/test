# -*- coding: utf-8 -*-
"""
比較答案與預測（兩檔 CSV 皆為欄位: text,label）
- 對齊方式：以 (text, 出現次序) 做鍵，能處理重複 text
- 輸出：
  1) 整體正確率
  2) 各「真實標籤」的錯誤數量
  3) 混淆矩陣（真實×預測）
"""
import argparse
import pandas as pd
from pathlib import Path

def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"找不到檔案：{p.resolve()}")
    # 嘗試處理 BOM/編碼問題
    try_encodings = ["utf-8-sig", "utf-8", "cp950"]
    last_err = None
    for enc in try_encodings:
        try:
            df = pd.read_csv(p, encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise SystemExit(f"讀取失敗（請檢查編碼與欄位名是否為 text,label）：{p}\n{last_err}")

def require_columns(df: pd.DataFrame, name: str):
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["text", "label"]
    for k in need:
        if k not in cols:
            raise SystemExit(f"{name} 缺少必要欄位 '{k}'；實際欄位：{list(df.columns)}")
    # 正規化欄位名稱（避免大小寫/空白異動）
    df.rename(columns={cols["text"]: "text", cols["label"]: "label"}, inplace=True)
    return df

def attach_occurrence_index(df: pd.DataFrame, key_col: str = "text") -> pd.DataFrame:
    # 為每個 text 加上出現次序 idx（從 0 起算），作為 disambiguation key
    return df.assign(_occ=df.groupby(key_col).cumcount())

def main():
    ap = argparse.ArgumentParser(description="計算整體正確率與各 label 錯誤數量（text,label 格式）")
    ap.add_argument("--answer", required=True, help="答案 CSV 檔 (text,label)")
    ap.add_argument("--pred", required=True, help="預測 CSV 檔 (text,label)")
    ap.add_argument("--show", type=int, default=10, help="列出前 N 筆錯誤示例（預設 10）")
    args = ap.parse_args()

    df_gold = read_csv(args.answer)
    df_pred = read_csv(args.pred)

    df_gold = require_columns(df_gold, "答案檔")
    df_pred = require_columns(df_pred, "預測檔")

    # 去除兩檔欄位可能的前後空白
    df_gold["text"] = df_gold["text"].astype(str).str.strip()
    df_gold["label"] = df_gold["label"].astype(str).str.strip()
    df_pred["text"] = df_pred["text"].astype(str).str.strip()
    df_pred["label"] = df_pred["label"].astype(str).str.strip()

    # 以 (text, 出現次序) 做對齊，可處理重複 text
    g = attach_occurrence_index(df_gold[["text", "label"]].copy())
    p = attach_occurrence_index(df_pred[["text", "label"]].copy())
    merged = g.merge(p, on=["text", "_occ"], how="inner", suffixes=("_true", "_pred"))

    if len(merged) == 0:
        raise SystemExit("兩檔無法對齊（text 完全不重疊？或編碼/清洗有問題）。請檢查輸入。")

    # 若對齊筆數 < 任一來源筆數，提示
    if len(merged) < len(df_gold) or len(merged) < len(df_pred):
        print(f"⚠️ 對齊筆數 {len(merged)} < 答案 {len(df_gold)} 或 預測 {len(df_pred)}，"
              f"可能有缺漏或 text 不一致（前後空白、大小寫、全半形）。")

    merged["correct"] = (merged["label_true"] == merged["label_pred"])

    # 整體正確率
    acc = merged["correct"].mean() if len(merged) else 0.0
    print(f"✅ 整體正確率：{acc:.4f}  ({merged['correct'].sum()}/{len(merged)})")

    # 各真實標籤的錯誤數量（= 該標籤的樣本中，被預測錯的數量）
    errs = merged[~merged["correct"]].groupby("label_true").size().sort_values(ascending=False)
    if errs.empty:
        print("🎉 所有樣本皆正確，無錯誤。")
    else:
        print("\n❌ 各真實標籤的錯誤數量（label_true -> count）")
        for lab, cnt in errs.items():
            print(f"  {lab}: {cnt}")

    # 混淆矩陣（真實 × 預測）
    print("\n🧭 混淆矩陣（真實 × 預測）")
    labels_sorted = sorted(set(merged["label_true"]) | set(merged["label_pred"]))
    cm = pd.crosstab(merged["label_true"], merged["label_pred"]).reindex(index=labels_sorted, columns=labels_sorted, fill_value=0)
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(cm)

    # 範例錯誤列出
    show_n = max(0, args.show)
    if show_n > 0:
        examples = merged[~merged["correct"]][["text", "label_true", "label_pred"]].head(show_n)
        if not examples.empty:
            print(f"\n🔍 前 {len(examples)} 筆錯誤示例：")
            for i, row in examples.iterrows():
                t = row["text"]
                # 只顯示前 80 字，避免過長
                t_disp = (t[:80] + "…") if len(t) > 80 else t
                print(f"- {t_disp}\n    true={row['label_true']}  pred={row['label_pred']}")

if __name__ == "__main__":
    main()
