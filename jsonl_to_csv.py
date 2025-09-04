import json
import csv

INPUT_FILE = "data/final_output.jsonl"   # 來源 JSONL 檔
OUTPUT_FILE = "data/final_output.csv"    # 輸出的 CSV 檔

with open(INPUT_FILE, "r", encoding="utf-8") as fin, open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["text", "label"])  # 寫入表頭

    for line in fin:
        if line.strip():  # 避免空行
            obj = json.loads(line)
            text = obj.get("text", "")
            label = obj.get("label", "")
            writer.writerow([text, label])

print(f"✅ 已經將 {INPUT_FILE} 轉換成 CSV，輸出到 {OUTPUT_FILE}")
