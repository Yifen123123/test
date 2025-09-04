import os
import json

# 你要輸入的資料夾路徑
input_folder = "output"

# 最後的結果會是一個 list，裡面存很多 dict
all_data = []

# 遍歷所有子資料夾與檔案
for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            
            # 讀取檔案內容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            # 這裡你可以依照需要處理 content 填入
            data = {
                "title": "",
                "date": "",
                "num": "",
                "general_subject": "",
                "sender_info": {
                    "address": "",
                    "phone": "",
                    "person": "",
                    "fax": "",
                    "email": "",
                },
                "receive_num": "",
                "receive_date": "",
                "raw_text": content  # 如果要保留原始文字可以加這行
            }

            all_data.append(data)

# 儲存成 JSON 檔案
with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("✅ 已經將所有結果存成 final_output.json")
