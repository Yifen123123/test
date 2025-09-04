all_data = []

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".txt"):
        file_path = os.path.join(FOLDER_PATH, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        extracted_info = extract_info_from_text(text)

        # 清理 general_subject
        extracted_info["general_subject"] = re.sub(
            r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_{|}~a-z\s]",
            "",
            extracted_info["general_subject"]
        )

        all_data.append(extracted_info)

# 儲存成 JSON 檔案
with open("final_output.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)

print("✅ 已經將結果存成 final_output.json")