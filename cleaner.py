import os, re, glob
from pii_detector import clean_line
from fake_data_loader import get_fake

INPUT_DIR  = "./input_txts"
OUTPUT_DIR = "./output_txts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 電話碎片後處理：合併標記 → 替換成一個假號碼
# ============================================================

def resolve_phone_fragments(text: str) -> str:
    """
    將連續出現的 [PHONE_FRAGMENT:xxx] 合併。
    若拼接後數字總長度 >= 8，視為一組完整電話 → 換成假號碼。
    若長度不足，維持標記（讓人工確認）。
    """
    # 抓出所有 fragment，記錄位置
    fragment_pattern = re.compile(r'\[PHONE_FRAGMENT:(\d+)\]')

    lines = text.split("\n")
    
    i = 0
    while i < len(lines):
        if "[PHONE_FRAGMENT:" not in lines[i]:
            i += 1
            continue

        # 找出連續含 fragment 的行
        group_indices = []
        accumulated = ""

        j = i
        while j < len(lines) and "[PHONE_FRAGMENT:" in lines[j]:
            for m in fragment_pattern.finditer(lines[j]):
                accumulated += m.group(1)
            group_indices.append(j)
            j += 1

        if len(accumulated) >= 8:
            # 合法電話長度 → 替換第一行，清空其他碎片行
            fake_phone = get_fake("phone")
            first_prefix = re.match(r'^([RLrl]\s*[:：]\s*)', lines[group_indices[0]])
            prefix = first_prefix.group(1) if first_prefix else ""
            lines[group_indices[0]] = f"{prefix}{fake_phone}"
            for idx in group_indices[1:]:
                lines[idx] = ""  # 清空碎片行
        else:
            # 長度不足，保留標記供人工確認
            pass

        i = j

    # 移除被清空的碎片行
    lines = [l for l in lines if l.strip() != ""]
    return "\n".join(lines)


# ============================================================
# 主流程
# ============================================================

def clean_file(filepath: str) -> str:
    with open(filepath, encoding="utf-8") as f:
        raw_lines = [l.rstrip("\n") for l in f.readlines()]

    # Step 1：每行做 regex + NER 替換，電話碎片先打標記
    cleaned_lines = [clean_line(line) for line in raw_lines]

    # Step 2：合併電話碎片標記，統一換假號碼
    full_text = "\n".join(cleaned_lines)
    full_text = resolve_phone_fragments(full_text)

    return full_text


def batch_clean():
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    print(f"共找到 {len(files)} 個檔案\n")

    for filepath in files:
        name = os.path.basename(filepath)
        result = clean_file(filepath)
        out_path = os.path.join(OUTPUT_DIR, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"✅  {name}  →  {out_path}")

    print("\n全部完成！")


if __name__ == "__main__":
    batch_clean()
