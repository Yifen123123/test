import csv

# 兩個檔案路徑
TRUE_FILE = "data/true.csv"       # 正確答案
PRED_FILE = "data/pred.csv"       # 模型預測

# 讀取 csv，回傳 {text: label}
def load_csv(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip()
            data[text] = label
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
