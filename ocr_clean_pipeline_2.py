#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import re
import unicodedata

# 目標：無論出現在句中或獨立成行，都刪掉下列字元（含常見簡繁體）
BLACKLIST = set(list("裝订訂装線线"))

# 移除的零寬/不可見字元（常見 OCR/複製貼上殘留）
INVISIBLES_RE = re.compile(
    r"[\u200B-\u200D\uFEFF\u2060\u00AD\u034F]"   # ZWSP, BOM, WJ, SHY, CGJ
)

# 只剩標點/符號/空白的行（清掉）
PUNCT_ONLY_RE = re.compile(r'^[\s\p{P}\p{S}]+$', re.UNICODE)

# Python 的 re 不支援 \p{...}，用簡易近似：去掉所有 Unicode 類別為字母或數字後是否為空
def is_punct_only(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    # 保留「字母/數字/漢字」；若去掉後沒東西，就視為純標點/空白
    kept = []
    for ch in s:
        cat = unicodedata.category(ch)
        # L* = Letter, N* = Number, Lo = Letter other(含漢字)
        if cat.startswith("L") or cat.startswith("N"):
            kept.append(ch)
    return len("".join(kept).strip()) == 0

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
    # 正規化到 NFKC，統一相容字型（全形/半形、相容漢字等）
    t = unicodedata.normalize("NFKC", t)
    # 去掉不可見字符
    t = INVISIBLES_RE.sub("", t)
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

def cleanup_lines(t: str) -> str:
    lines = t.splitlines()
    cleaned = []
    for ln in lines:
        # 行尾/行首空白收斂
        s = ln.strip()
        if not s:
            # 空行直接略過
            continue
        if is_punct_only(s):
            # 只有標點/符號的行略過（處理 .. . .. ... 這種）
            continue
        cleaned.append(s)
    return "\n".join(cleaned)

def main():
    ap = argparse.ArgumentParser(description="刪除『裝』『訂』『線』（含簡繁體變體），並清理空白/純標點行。")
    ap.add_argument("input", help="輸入 .txt 檔路徑")
    ap.add_argument("-o", "--output", help="輸出檔名（預設 input.clean.txt）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[錯誤] 找不到檔案：{in_path}", file=sys.stderr)
        sys.exit(1)

    raw, used_enc = read_text_any_encoding(in_path)
    print(f"[INFO] 讀取編碼：{used_enc}")

    # 步驟 1：Unicode 正規化 + 去不可見字元
    t = normalize_text(raw)

    # 步驟 2：刪除黑名單字元（裝/訂/線 + 简体）
    t, removed = remove_blacklist_chars(t)
    print(f"[INFO] 已移除目標字元數量：{removed}")

    # 步驟 3：清掉因此變成空白/純標點的行
    t = cleanup_lines(t)

    out_path = Path(args.output) if args.output else in_path.with_suffix(".clean.txt")
    out_path.write_text(t, encoding="utf-8")
    print(f"[完成] 已輸出：{out_path}")

if __name__ == "__main__":
    main()
