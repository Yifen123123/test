import re
from fake_data_loader import get_fake

# ============================================================
# 一、安裝與載入 CKIP NER（只在第一次呼叫時初始化）
# pip install ckip-transformers transformers torch
# ============================================================
_ner_pipeline = None

def get_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        print("載入 CKIP NER 模型（首次需下載，約 400MB）...")
        from ckip_transformers.nlp import CkipNerChunker
        _ner_pipeline = CkipNerChunker(model="bert-base")
    return _ner_pipeline


# ============================================================
# 二、NER 替換（姓名、地址、組織）
# ============================================================

NER_CATEGORY_MAP = {
    "PERSON":   "name",
    "GPE":      "address",   # 地理政治實體（縣市）
    "LOC":      "address",   # 地點
    "FAC":      "address",   # 設施（地址中的建物）
}

def replace_by_ner(text: str) -> str:
    """用 CKIP NER 偵測姓名與地址，替換成假資料"""
    ner = get_ner()
    # CKIP 接受 list of sentences
    results = ner([text], use_delim=False)
    entities = results[0]  # [(word, ner_tag), ...]

    # 由後往前替換，避免位移問題
    replacements = []
    for entity, ner_tag in entities:
        base_tag = ner_tag.split("-")[-1]  # 去除 B-/I- 前綴
        if base_tag in NER_CATEGORY_MAP:
            fake_cat = NER_CATEGORY_MAP[base_tag]
            replacements.append((entity, get_fake(fake_cat)))

    # 去重後替換（同一個詞只換一次）
    seen = set()
    for original, fake in replacements:
        if original not in seen:
            text = text.replace(original, fake)
            seen.add(original)

    return text


# ============================================================
# 三、Regex 替換（結構化個資：電話、身分證、生日、email）
# ============================================================

def replace_email(text: str) -> str:
    return re.sub(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        lambda m: get_fake("email"),
        text
    )

def replace_id(text: str) -> str:
    # 身分證：1英文 + 1[12] + 8數字
    return re.sub(
        r'\b[A-Z][12]\d{8}\b',
        lambda m: get_fake("id"),
        text
    )

def replace_birthday(text: str) -> str:
    # 支援：1990/01/01, 79年1月1日, 19900101
    return re.sub(
        r'\b(?:民國\s*)?\d{2,4}\s*[年/\-]\s*\d{1,2}\s*[月/\-]\s*\d{1,2}\s*[日]?\b',
        lambda m: get_fake("birthday"),
        text
    )

def replace_phone_single_line(text: str) -> str:
    """單行內完整電話號碼 → 直接換假號碼"""
    return re.sub(
        r'0[2-9]\d[\s\-]?\d{3,4}[\s\-]?\d{3,4}',
        lambda m: get_fake("phone"),
        text
    )

def tag_phone_fragment(line: str) -> str:
    """
    單行內只有純數字碎片（疑似電話片段）→ 標記為 [PHONE_FRAGMENT]
    判斷條件：去除 R:/L: 前綴後，內容全為數字（允許空格/連字號），且長度 2~6 碼
    """
    prefix_match = re.match(r'^([RLrl]\s*[:：]\s*)(.*)', line)
    if not prefix_match:
        return line

    prefix, content = prefix_match.group(1), prefix_match.group(2).strip()
    clean = re.sub(r'[\s\-]', '', content)

    # 純數字 且 長度符合碎片範圍（2~6碼）
    if re.fullmatch(r'\d{2,6}', clean):
        return f"{prefix}[PHONE_FRAGMENT:{clean}]"

    return line


# ============================================================
# 四、整合：單行完整清洗流程
# ============================================================

def clean_line(line: str) -> str:
    line = replace_email(line)
    line = replace_id(line)
    line = replace_birthday(line)
    line = replace_phone_single_line(line)
    line = tag_phone_fragment(line)
    line = replace_by_ner(line)
    return line
