#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import unicodedata
import re

# 目標：無論出現在句中或獨立成行，都刪掉下列字元（含常見簡繁體）
BLACKLIST = set(list("裝订訂装線线"))

# 常見不可見字元（零寬空白等）
INVISIBLE_CHARS = {
    "\u200B",  # ZWSP
    "\u200C",  # ZWNJ
    "\u200D",  # ZWJ
    "\uFEFF",  # BOM
    "\u2060",  # WJ
    "\u00AD",  # SHY
    "\u034F",  # CGJ
}

def read_text_any_encoding(path: Path) -> tuple[str, str]:
    encodings = [
        "utf-8-sig", "utf-8",
        "cp950", "big5",
        "gb18030",
        "utf-16", "utf-16le", "utf-16be",
    ]
    for enc in encodings:
        try:
            txt = path.read_text(encoding=enc)
            return txt, enc
        except Exception:
            continue
    # 最後手段：忽略錯誤讀 UTF-8
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt, "utf-8(ignore)"

def normalize_text(t: str) -> str:
    # 正規化（全半形等統一）
    t = unicodedata.normalize("NFKC", t)
    # 去掉不可見字元
    if any(ch in t for ch in INVISIBLE_CHARS):
        for ch in INVISIBLE_CHARS:
            t = t.replace(ch, "")
    return t

def remove_blacklist_chars(t: str) -> tuple[str, int]:
    removed = 0
    out_chars = []
    for ch in t:
        if ch in BLACKLIST:
            removed += 1
            continue
        out_chars.append(ch)
    return "".join(out_chars), removed

def is_punct_only_line(s: str) -> bool:
    """
    若一行去掉所有「字母/數字」後，什麼都不剩，就視為只有標點/符號/空白。
    （用 unicodedata.category，避免 \\p 語法）
    """
    s = s.strip()
    if not s:
        return True
    kept = []
    for ch in s:
        cat = unicodedata.category(ch)  # 'L*' 字母, 'N*' 數字（Lo 含漢字）
        if cat.startswith("L") or cat.startswith("N"):
            kept.append(ch)
    return len("".join(kept).strip()) == 0

def remove_leading_dots(t: str) -> str:
    """
    移除每一行「行首的點串」：
    例：".abc" → "abc"、"..  xyz" → "xyz"、"...abc" → "abc"
    也容忍行首空白： "   ...abc" → "abc"
    """
    lines = t.splitlines()
    new_lines = [re.sub(r'^\s*\.+\s*', '', ln) for ln in lines]
    return "\n".join(new_lines)

def cleanup_lines(t: str) -> str:
    """
    清掉空行與純標點/符號行（例如 ..、...、｜、／ 等）。
    """
    lines = t.splitlines()
    cleaned = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if is_punct_only_line(s):
            continue
        cleaned.append(s)
    return "\n".join(cleaned)

def main():
    ap = argparse.ArgumentParser(description="刪除『裝』『訂』『線』、行首點串（. .. ...）、並清理空白/純標點行（純標準庫版）。")
    ap.add_argument("input", help="輸入 .txt 檔路徑")
    ap.add_argument("-o", "--output", help="輸出檔名（預設 input.clean.txt）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[錯誤] 找不到檔案：{in_path}", file=sys.stderr)
        sys.exit(1)

    raw, used_enc = read_text_any_encoding(in_path)
    print(f"[INFO] 讀取編碼：{used_enc}")

    # 1) 正規化 + 去不可見字元
    t = normalize_text(raw)

    # 2) 刪除黑名單字元（裝/訂/線，含簡體） 
    t, removed = remove_blacklist_chars(t)
    print(f"[INFO] 已移除目標字元數量：{removed}")

    # 3) 清掉每行行首的點（. .. ...）
    t = remove_leading_dots(t)

    # 4) 移除空行與純標點/符號行
    t = cleanup_lines(t)

    out_path = Path(args.output) if args.output else in_path.with_suffix(".clean.txt")
    out_path.write_text(t, encoding="utf-8")
    print(f"[完成] 已輸出：{out_path}")

if __name__ == "__main__":
    main()
