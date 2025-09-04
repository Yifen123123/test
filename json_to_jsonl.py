import json
import os

INPUT_FILE = "final_output.json"      # 你的來源 JSON 檔
OUTPUT_FILE = "data/final_output.jsonl"

# 確保輸出資料夾存在
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# 讀取原始 JSON
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 寫入 JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in data:
        new_item = {
            "label": item.get("label", ""),
            "text": item.get("general_subject", "")
        }
        f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

print(f"✅ 已經將 {INPUT_FILE} 轉換成 JSONL，輸出到 {OUTPUT_FILE}")
