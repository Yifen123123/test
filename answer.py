import csv

TRUE_FILE = "data/true.csv"   # 正確答案
PRED_FILE = "data/pred.csv"   # 預測結果

def load_csv(path):
    data = {}
    with open(path, "r", encoding="utf-8-sig") as f:  # 用 utf-8-sig 可避開 BOM 問題
        reader = csv.DictReader(f)
        print(f"讀取 {path} 的欄位名稱:", reader.fieldnames)  # debug 用
        for row in reader:
            # 自動找欄位（容錯）
            text = row.get("text") or row.get("Text") or row.get("TEXT")
            label = row.get("label") or row.get("Label") or row.get("LABEL")
            if text is None or label is None:
                raise ValueError(f"⚠️ {path} 缺少 text 或 label 欄位，實際欄位: {reader.fieldnames}")
            data[text.strip()] = label.strip()
    return data

true_data = load_csv(TRUE_FILE)
pred_data = load_csv(PRED_FILE)

# 計算正確率
total = 0
correct = 0
for text, true_label in true_data.items():
    if text in pred_data:
        total += 1
        if pred_data[text] == true_label:
            correct += 1

accuracy = correct / total if total > 0 else 0.0

print(f"總數: {total}")
print(f"正確數: {correct}")
print(f"正確率: {accuracy:.4f}")
