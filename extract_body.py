# -*- coding: utf-8 -*-
"""
extract_body.py
----------------
從 OCR 文字檔（.txt）抽取臺灣制式公文欄位，重點擷取正文（內文）。
支援批次處理：遍歷 input_root 下所有子資料夾與 .txt 檔，輸出到 output_root（每類別一個 .jsonl）。

使用方式：
  基本：
    python extract_body.py
  指定輸入/輸出資料夾：
    python extract_body.py --input-root output --output-root parsed
  僅處理特定子資料夾（可多個）：
    python extract_body.py --folders 保單查詢 通知函
  僅處理檔名包含關鍵字的 .txt：
    python extract_body.py --filename-contains 註記

輸出：
  parsed/保單查詢.jsonl
  parsed/通知函.jsonl
  ...
每行一份文件的 JSON 結果（UTF-8, ensure_ascii=False）
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# =========================
# 1) 欄位別名與容錯正規化
# =========================

FIELD_ALIASES = {
    "recipient":  ["受文者", "受文機關", "受文單位"],
    "doc_no":     ["發文字號", "文號", "案號", "來文字號"],
    "date":       ["發文日期", "日期", "中華民國"],
    "priority":   ["速別"],
    "security":   ["密等"],
    "subject":    ["主旨"],
    "body":       ["說明", "內文", "正文", "本文"],
    "attachment": ["附件"],
    "cc":         ["副本", "正本", "抄送"],
    "contact":    ["承辦", "承辦人", "聯絡", "聯絡電話", "連絡電話"],
}

# OCR 常見錯字 / 全半形 / 冒號 / 分隔符 正規化
CANONICAL_REPLACEMENTS = [
    # 常見誤辨
    ("王旨", "主旨"), ("圭旨", "主旨"),
    ("說朋", "說明"), ("說眀", "說明"),
    # 冒號與空白
    ("：", ":"), ("﹕", ":"), ("︰", ":"), ("：", ":"),
    ("　", " "), ("﻿", ""), ("\ufeff", ""),
    # 破折號類
    ("－", "-"), ("—", "-"), ("–", "-"),
]


def normalize_text(raw: str) -> str:
    """一般清洗：錯字替換、換行統一、頁眉/頁腳/行號/分隔線清理。"""
    text = raw
    for a, b in CANONICAL_REPLACEMENTS:
        text = text.replace(a, b)

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = text.split("\n")
    cleaned = []
    for ln in lines:
        s = ln.strip()
        # 去除只有頁碼或分隔線的行
        if re.fullmatch(r"-{3,}|_{3,}|=+|~+|\d+/\d+|第\d+頁", s):
            continue
        # 去掉每行開頭行號（若有）
        s = re.sub(r"^\s*\(?\d{1,3}\)?\s+", "", s)
        cleaned.append(s)
    text = "\n".join(cleaned)
    return text


def build_field_regex() -> re.Pattern:
    """動態建 regex 以偵測欄位名行：支援『欄位: 內容』或下一行開始內容。"""
    names: List[str] = []
    for _, alias in FIELD_ALIASES.items():
        names.extend(alias)
    names = sorted(set(names), key=lambda x: -len(x))  # 長詞優先

    pattern = r"^(?P<field>(" + "|".join(map(re.escape, names)) + r"))\s*:?\s*(?P<after>.*)$"
    return re.compile(pattern, flags=re.MULTILINE)


FIELD_PATTERN = build_field_regex()


# =========================
# 2) 核心抽取
# =========================

def split_sections(text: str) -> Dict[str, str]:
    """以欄位錨點切段，回傳『命中的中文欄位名』→『內容』的對照表。"""
    matches = list(FIELD_PATTERN.finditer(text))
    if not matches:
        return {}

    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        field_name = m.group("field")
        after = m.group("after").strip()
        block = (after + "\n" + text[start:end]).strip() if after else text[start:end].strip()
        block = block.strip()
        # 同名欄位多次出現就串接
        if field_name in sections and sections[field_name]:
            sections[field_name] = (sections[field_name] + "\n" + block).strip()
        else:
            sections[field_name] = block
    return sections


def canonicalize_keys(sections: Dict[str, str]) -> Dict[str, str]:
    """將中文欄位名轉成標準鍵（recipient/doc_no/date/subject/body/...）。"""
    out: Dict[str, str] = {}
    for canon, aliases in FIELD_ALIASES.items():
        for a in aliases:
            if a in sections and sections[a].strip():
                out[canon] = sections[a].strip()
                break
    return out


def _span_of_field(field_cn: str, text: str) -> Optional[Tuple[int, int]]:
    """取得文本中某中文欄位名對應的內容範圍（行尾到下一欄位或文末）。"""
    pat = re.compile(r"^" + re.escape(field_cn) + r"\s*:?\s*(.*)$", flags=re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    start = m.end()
    nxt = FIELD_PATTERN.search(text, pos=start)
    end = nxt.start() if nxt else len(text)
    return (start, end)


def _middle_block(text: str) -> str:
    """保底：取中間 60% 內容作為正文候選（去掉頭尾噪音）。"""
    n = len(text)
    if n == 0:
        return ""
    return text[int(n * 0.2): int(n * 0.8)]


def heuristic_body(text: str, sections_raw: Dict[str, str], sections: Dict[str, str]) -> str:
    """
    正文回退策略：
      1) 若有「說明/內文/正文/本文」則直接取。
      2) 否則取「主旨」之後 → 「附件/副本/承辦」之前。
      3) 再不行：偵測條列起手式（「一、二、三、」）。
      4) 最後保底取中段。
    """
    if "body" in sections and sections["body"].strip():
        return sections["body"].strip()

    tail_markers = FIELD_ALIASES["attachment"] + FIELD_ALIASES["cc"] + FIELD_ALIASES["contact"]

    # 有主旨時：主旨之後 → 最近尾標之前
    subject_span = _span_of_field("主旨", text)
    if subject_span:
        start = subject_span[1]
        tail_positions = []
        for t in tail_markers:
            sp = _span_of_field(t, text)
            if sp and sp[0] > start:
                tail_positions.append(sp[0])
        end = min(tail_positions) if tail_positions else len(text)
        body = text[start:end].strip()
        # 砍掉意外殘留的欄位行
        body = re.sub(FIELD_PATTERN, "", body).strip()
        if body:
            return body

    # 沒主旨或上面抓不到：找條列起手式
    m = re.search(r"^[（(]?(一|二|三|四|五|六|七|八|九|十)[)）]?[、.．]", text, flags=re.MULTILINE)
    if m:
        return text[m.start():].strip()

    # 保底
    return _middle_block(text).strip()


def extract_body_from_txt(txt_path: str, encoding: str = "utf-8") -> Dict[str, object]:
    """讀取單一 .txt 檔並抽取欄位。"""
    with open(txt_path, "r", encoding=encoding, errors="ignore") as f:
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


# =========================
# 3) 批次處理與 CLI
# =========================

def process_all(
    input_root: Path,
    output_root: Path,
    only_folders: Optional[List[str]] = None,
    filename_contains: Optional[str] = None,
    encoding: str = "utf-8"
) -> None:
    """
    遍歷 input_root 下所有子資料夾與 .txt，輸出每一類別到 output_root/<類別>.jsonl
    - only_folders: 限定只處理這些子資料夾（名稱完全比對）
    - filename_contains: 只處理檔名中包含此關鍵字的 .txt
    """
    output_root.mkdir(exist_ok=True, parents=True)

    # 收集子資料夾
    folders = [p for p in input_root.iterdir() if p.is_dir()]
    if only_folders:
        target_set = set(only_folders)
        folders = [p for p in folders if p.name in target_set]

    if not folders:
        print(f"⚠️ 找不到子資料夾可處理（root: {input_root}）")
        return

    for folder in sorted(folders, key=lambda p: p.name):
        category = folder.name
        out_file = output_root / f"{category}.jsonl"

        txt_files = sorted(folder.glob("*.txt"))
        if filename_contains:
            txt_files = [p for p in txt_files if filename_contains in p.name]

        if not txt_files:
            print(f"ℹ️ 類別「{category}」沒有符合條件的 .txt 檔，略過。")
            continue

        count_ok, count_err = 0, 0
        with open(out_file, "w", encoding="utf-8") as fout:
            for txt_path in txt_files:
                try:
                    result = extract_body_from_txt(str(txt_path), encoding=encoding)
                    result["filename"] = txt_path.name
                    result["category"] = category
                    # 輸出 JSON 一行
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    count_ok += 1
                except Exception as e:
                    print(f"❌ 處理失敗：{txt_path} -> {e}")
                    count_err += 1
        print(f"✅ 類別「{category}」完成：成功 {count_ok} 筆，失敗 {count_err} 筆 -> {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽取 OCR 公文內文與欄位（批次處理資料夾）"
    )
    parser.add_argument(
        "--input-root", type=str, default="output",
        help="輸入根資料夾（底下每個子資料夾是一種類別）"
    )
    parser.add_argument(
        "--output-root", type=str, default="parsed",
        help="輸出根資料夾（每類輸出一個 <類別>.jsonl）"
    )
    parser.add_argument(
        "--folders", nargs="*", default=None,
        help="僅處理指定子資料夾名稱（可多個），預設處理全部"
    )
    parser.add_argument(
        "--filename-contains", type=str, default=None,
        help="僅處理檔名包含此關鍵字的 .txt"
    )
    parser.add_argument(
        "--encoding", type=str, default="utf-8",
        help="輸入檔案編碼（預設 utf-8）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    if not input_root.exists():
        print(f"❌ 找不到輸入資料夾：{input_root.resolve()}")
        return

    print(f"🚀 開始處理：input_root={input_root.resolve()}  ->  output_root={output_root.resolve()}")
    if args.folders:
        print(f"   只處理子資料夾：{', '.join(args.folders)}")
    if args.filename_contains:
        print(f"   只處理檔名包含：{args.filename_contains}")
    process_all(
        input_root=input_root,
        output_root=output_root,
        only_folders=args.folders,
        filename_contains=args.filename_contains,
        encoding=args.encoding
    )
    print("🎉 全部完成。")


if __name__ == "__main__":
    main()
