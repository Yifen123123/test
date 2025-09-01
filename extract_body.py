import re
from typing import Dict, Tuple, List, Optional

# ---- 1) 常見欄位錨點（依常見程度與解析順序排列）----
FIELD_ALIASES = {
    "recipient":  ["受文者", "受文機關", "受文單位"],
    "doc_no":     ["發文字號", "文號", "案號"],
    "date":       ["發文日期", "日期", "中華民國"],
    "priority":   ["速別"],
    "security":   ["密等"],
    "subject":    ["主旨"],
    # 內文常見別名（正規抽取主要目標）
    "body":       ["說明", "內文", "正文", "本文"],
    "attachment": ["附件"],
    "cc":         ["副本", "正本", "抄送"],
    "contact":    ["承辦", "聯絡", "聯絡電話", "連絡電話"],
}

# ---- 2) OCR 常見錯字 / 全半形 正規化替換表 ----
CANONICAL_REPLACEMENTS = [
    ("王旨", "主旨"), ("圭旨", "主旨"),
    ("說朋", "說明"), ("說眀", "說明"),
    ("吿", "告"), ("氐", "氏"),
    # 標點 / 全半形/破折號/冒號/空白
    ("：", ":"), ("﹕", ":"), ("︰", ":"), ("：", ":"),
    ("　", " "), ("﻿", ""), ("\ufeff", ""),  # BOM/奇怪空白
    ("－", "-"), ("—", "-"), ("–", "-"),
]

# ---- 3) 前處理：清洗、合併斷行、移除頁眉頁腳線號 ----
def normalize_text(raw: str) -> str:
    text = raw
    for a, b in CANONICAL_REPLACEMENTS:
        text = text.replace(a, b)
    # 統一換行
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除明顯的頁碼/分隔線（啟發式）
    lines = text.split("\n")
    cleaned = []
    for ln in lines:
        s = ln.strip()
        # 去除只有頁碼或分隔線的行
        if re.fullmatch(r"-{3,}|_{3,}|= {0,}|=+|~+|\d+/\d+|第\d+頁", s):
            continue
        # 去掉每行行號（若有）
        s = re.sub(r"^\s*\(?\d{1,3}\)?\s+", "", s)
        cleaned.append(s)
    text = "\n".join(cleaned)
    # 合併「行尾有全形/半形逗號、頓號、分號」但下一行不是新欄位名的斷行
    text = re.sub(r"(?<!:)\n(?=[^\n])", "\n", text)  # 保留自然段，先不做太激進的合併
    return text

# ---- 4) 動態建出欄位名的 regex（容許「欄位名: 內容」或「欄位名」獨立成行）----
def build_field_regex() -> re.Pattern:
    # 將所有別名扁平化，並依長度排序（避免短詞搶先匹配）
    names: List[str] = []
    for _, alias in FIELD_ALIASES.items():
        names.extend(alias)
    names = sorted(set(names), key=lambda x: -len(x))

    # 允許「欄位名」後面直接冒號或空白；允許同行有內容或下一行開始內容
    # 用 (?P<field>...) 捕捉欄位名；(?P<after>...) 捕捉同行即時內容
    pattern = r"^(?P<field>(" + "|".join(map(re.escape, names)) + r"))\s*:?\s*(?P<after>.*)$"
    return re.compile(pattern, flags=re.MULTILINE)

FIELD_PATTERN = build_field_regex()

# ---- 5) 主抽取：切段成欄位 -> 內容的 map ----
def split_sections(text: str) -> Dict[str, str]:
    """
    先用欄位名錨點切段。回傳 key 為欄位中文名（實際命中字串），值為該欄位的全文內容。
    """
    matches = list(FIELD_PATTERN.finditer(text))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        field_name = m.group("field")
        # 內容包含「同行 after」+ 下一段落區間
        after = m.group("after").strip()
        block = (after + "\n" + text[start:end]).strip() if after else text[start:end].strip()
        # 清掉下一個欄位名前殘留的空白
        block = block.strip()
        # 若同名欄位多次出現，串接
        sections[field_name] = (sections.get(field_name, "") + ("\n" if sections.get(field_name) else "") + block).strip()
    return sections

# ---- 6) 將中文欄位名對應到標準鍵（recipient/doc_no/date/subject/body/...）----
def canonicalize_keys(sections: Dict[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for canon, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if a in sections and sections[a].strip():
                out[canon] = sections[a].strip()
                break
    # 保留原始 subject/body 等
    return out

# ---- 7) 若沒命中「body」：啟發式回退 ----
def heuristic_body(text: str, sections_raw: Dict[str, str], sections: Dict[str, str]) -> str:
    # 優先：有「說明/內文/正文/本文」就直接回傳
    if "body" in sections and sections["body"].strip():
        return sections["body"].strip()

    # 次選：若有主旨，則「主旨」之後到下一個尾欄位（附件/副本/正本/承辦…）視為正文
    tail_markers = FIELD_ALIASES["attachment"] + FIELD_ALIASES["cc"] + FIELD_ALIASES["contact"]
    # 找主旨段
    subject_span = _span_of_field("主旨", text)
    if not subject_span:
        # 沒主旨：抓「條列起手式」之後的段落（如「一、」「二、」）
        m = re.search(r"^[（(]?(一|二|三|四|五|六|七|八|九|十)[)）]?[、.．]", text, flags=re.MULTILINE)
        if m:
            start = m.start()
            return text[start:].strip()
        # 退而求其次：整份文本去除頭尾噪音，取中段
        return _middle_block(text).strip()

    # 有主旨：找主旨後的最近尾標
    start = subject_span[1]
    # 找離 start 最近的任何尾欄位位置
    tail_positions = []
    for t in tail_markers:
        sp = _span_of_field(t, text)
        if sp and sp[0] > start:
            tail_positions.append(sp[0])
    end = min(tail_positions) if tail_positions else len(text)
    body = text[start:end].strip()
    # 避免把下一個欄位名含進來：砍掉開頭殘留欄位名行
    body = re.sub(FIELD_PATTERN, "", body).strip()
    return body

def _span_of_field(field_cn: str, text: str) -> Optional[Tuple[int, int]]:
    pat = re.compile(r"^" + re.escape(field_cn) + r"\s*:?\s*(.*)$", flags=re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    # 從該行結束到下一個欄位或文末
    start = m.end()
    # 找下一欄位
    nxt = FIELD_PATTERN.search(text, pos=start)
    end = nxt.start() if nxt else len(text)
    return (start, end)

def _middle_block(text: str) -> str:
    # 簡單抓中間 60% 作為內文候選（去掉前 20%、後 20%）
    n = len(text)
    return text[int(n*0.2): int(n*0.8)]

# ---- 8) 主入口：讀檔 -> 抽取 ----
def extract_body_from_txt(txt_path: str) -> Dict[str, str]:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    norm = normalize_text(raw)
    sections_raw = split_sections(norm)
    sections = canonicalize_keys(sections_raw)
    body = heuristic_body(norm, sections_raw, sections)
    return {
        "subject": sections.get("subject", ""),
        "body": body,
        "attachment": sections.get("attachment", ""),
        "meta": {
            "recipient": sections.get("recipient", ""),
            "doc_no": sections.get("doc_no", ""),
            "date": sections.get("date", ""),
            "priority": sections.get("priority", ""),
            "security": sections.get("security", ""),
        }
    }

# ---- 9) 簡單測試（將 path 換成你的 OCR 文字檔）----
if __name__ == "__main__":
    # demo_path = "sample_ocr.txt"
    # print(extract_body_from_txt(demo_path))
    pass
