FOLDER_PATH = "next_folder"   # 你要換的資料夾
OUTPUT_FILE = "final_output.json"

# Step 1: 如果 JSON 檔存在，就讀進來
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        all_data = json.load(f)
else:
    all_data = []

# Step 2: 處理新資料夾
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".txt"):
        file_path = os.path.join(FOLDER_PATH, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        extracted_info = extract_info_from_text(text)
        extracted_info["general_subject"] = re.sub(
            r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_{|}~a-z\s]",
            "",
            extracted_info["general_subject"]
        )
        all_data.append(extracted_info)

# Step 3: 存回 JSON 檔（舊+新）
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print(f"✅ 已經把 {FOLDER_PATH} 的資料追加到 {OUTPUT_FILE}")