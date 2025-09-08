# eval_test_dual.py
import argparse
import pandas as pd
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def load_preds(path, name):
    df = pd.read_csv(path)
    need = {"doc_id","pred"}
    if not need.issubset(df.columns):
        raise SystemExit(f"[{name}] 缺欄位 {need}：{path}")
    return df[["doc_id","pred"]].rename(columns={"pred": f"pred_{name}"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True, help="含 doc_id, label, built_text, built_text_masked")
    ap.add_argument("--pred_masked", required=True, help="用 built_text_masked 推論出的預測檔")
    ap.add_argument("--pred_unmasked", required=True, help="用 built_text 推論出的預測檔（對照）")
    args = ap.parse_args()

    test = pd.read_csv(args.test_csv)
    need_cols = {"doc_id","label"}
    if not need_cols.issubset(test.columns):
        raise SystemExit(f"[test] 缺欄位 {need_cols}")

    pm = load_preds(args.pred_masked, "masked")
    pu = load_preds(args.pred_unmasked, "unmasked")

    mix = test[["doc_id","label"]].merge(pm, on="doc_id", how="left").merge(pu, on="doc_id", how="left")

    # A) 遮蔽版評分
    y_true = mix["label"].astype(str)
    y_m = mix["pred_masked"].astype(str)
    y_u = mix["pred_unmasked"].astype(str)

    print("==== 遮蔽版（masked features）====")
    print(classification_report(y_true, y_m, digits=4))
    print("macro-F1(masked) =", f1_score(y_true, y_m, average="macro"))
    print()

    print("==== 未遮蔽對照（unmasked features）====")
    print(classification_report(y_true, y_u, digits=4))
    print("macro-F1(unmasked) =", f1_score(y_true, y_u, average="macro"))
    print()

    # 類別別差異（看哪些類別靠關鍵詞撐分數）
    by_cls = (mix.assign(ok_m=(y_true==y_m), ok_u=(y_true==y_u))
                 .groupby("label")[["ok_m","ok_u"]].mean().reset_index())
    by_cls["delta(u-m)"] = by_cls["ok_u"] - by_cls["ok_m"]
    print("==== 各類別 準確率比較（unmasked - masked）====")
    print(by_cls.sort_values("delta(u-m)", ascending=False).to_string(index=False))

    # 混淆矩陣（遮蔽版）
    labs = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_m, labels=labs)
    print("\n==== 混淆矩陣（masked）====")
    print("labels:", labs)
    print(cm)

if __name__ == "__main__":
    main()
