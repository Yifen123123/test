#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import unicodedata
import re

# ====================== 單檔清洗邏輯（與你前面版本對齊） ======================

BLACKLIST = {"裝", "订", "訂", "装", "線", "线"}  # 逐字刪除
INVISIBLE_CHARS = {"\u200B", "\u200C", "\u200D", "\uFEFF", "\u2060", "\u00AD", "\u034F"}
CJK_RANGE = r"\u4e00-\u9fff"
CJK_PUNCT = r"。，、：；（）()《》【】「」『』—－─"

def read_text_any_encoding(path: Path) -> tuple[str, str]:
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5", "gb18030", "utf-16", "utf-16le", "utf-16be"]
    for enc in encodings:
        try:
            txt = path.read_text(encoding=enc)
            return txt, enc
        except Exception:
            continue
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return txt, "utf-8(ignore)"

def normalize_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    for ch in INVISIBLE_CHARS:
        if ch in t:
            t = t.replace(ch, "")
    return t

def remove_blacklist_chars(t: str) -> tuple[str, int]:
    removed = 0
    out = []
    for ch in t:
        if ch in BLACKLIST:
            removed += 1
        else:
            out.append(ch)
    return "".join(out), removed

def is_punct_only_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    kept = []
    for ch in s:
        cat = unicodedata.category(ch)  # 'L*' 字母, 'N*' 數字（Lo 包含漢字）
        if cat.startswith("L") or cat.startswith("N"):
            kept.append(ch)
    return len("".join(kept).strip()) == 0

def remove_leading_dots(t: str) -> str:
    # 移除行首的 . ． …（允許行首空白）
    return "\n".join(re.sub(r'^\s*[\.．…]+\s*', '', ln) for ln in t.splitlines())

def remove_page_markers(t: str) -> str:
    # 刪「第 X 頁 … 共 Y 頁」整行頁碼標記
    pat = re.compile(r"^第\s*\d+\s*頁\s*[,，、]?\s*共\s*\d+\s*頁$")
    out = []
    for ln in t.splitlines():
        if pat.match(ln.strip()):
            continue
        out.append(ln)
    return "\n".join(out)

def clean_inline_noise_once(s: str) -> str:
    # 1) 連續雜訊符號塊（兩個以上，非 \w、空白、常見中文標點）
    s = re.sub(r"[^\w\s" + CJK_PUNCT + r"]{2,}", " ", s)
    # 2) 常見雜訊符號（任意長度）
    s = re.sub(r"[\^ˇ＾~`\\/|<>]+", "", s)
    # 3) 英字 + 雜訊(+空白) + 英字（如 J<f、V <f）
    s = re.sub(r"\b[A-Za-z]\s*[^0-9A-Za-z\s" + CJK_RANGE + r"]{1,3}\s*[A-Za-z]\b", "", s)
    # 4) 孤立單一英文字母（左右為中文/標點/空白或邊界；如 '。r 章'）
    s = re.sub(
        rf"(?:(?<=^)|(?<=[\s{CJK_RANGE}{CJK_PUNCT}]))[A-Za-z](?=(?:$|[\s{CJK_RANGE}{CJK_PUNCT}]))",
        "",
        s
    )
    # 5) 收斂空白
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def clean_inline_noise(t: str, max_pass: int = 5) -> str:
    lines = t.splitlines()
    out = []
    for ln in lines:
        prev = ln
        for _ in range(max_pass):
            cur = clean_inline_noise_once(prev)
            if cur == prev:
                break
            prev = cur
        out.append(prev)
    return "\n".join(out)

def cleanup_lines(t: str) -> str:
    out = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        if is_punct_only_line(s):
            continue
        out.append(s)
    return "\n".join(out)

def process_text(raw: str, verbose: bool = False) -> str:
    t = normalize_text(raw)
    t, removed = remove_blacklist_chars(t)
    if verbose:
        print(f"[INFO] 移除『裝/訂/線』等字元：{removed}")
    t = remove_leading_dots(t)
    t = remove_page_markers(t)
    t = clean_inline_noise(t, max_pass=5)
    t = cleanup_lines(t)
    return t

# ====================== 批次處理（遞迴掃描 + 鏡像輸出） ======================

def process_tree(input_root: Path, output_root: Path | None, suffix: str = "", inplace: bool = False, verbose: bool = True) -> tuple[int, int]:
    """
    遞迴處理 input_root 底下所有 .txt。
    - 若 inplace=True：覆寫原檔。
    - 否則：輸出到 output_root，維持相同相對路徑；可用 suffix 加檔名尾綴（在副檔名前）。
    回傳: (成功數, 失敗數)
    """
    if not input_root.exists():
        raise FileNotFoundError(f"找不到資料夾：{input_root}")

    if not inplace and output_root is None:
        raise ValueError("未指定 output_root，也未使用 --inplace")

    success = 0
    fail = 0

    for in_file in input_root.rglob("*.txt"):
        try:
            raw, used_enc = read_text_any_encoding(in_file)
            if verbose:
                rel = in_file.relative_to(input_root)
                print(f"[處理] {rel}  (encoding={used_enc})")

            cleaned = process_text(raw, verbose=False)

            if inplace:
                out_path = in_file
            else:
                rel = in_file.relative_to(input_root)
                out_dir = (output_root / rel.parent)
                out_dir.mkdir(parents=True, exist_ok=True)
                if suffix:
                    out_name = rel.stem + suffix + rel.suffix
                else:
                    out_name = rel.name
                out_path = out_dir / out_name

            out_path.write_text(cleaned, encoding="utf-8")
            success += 1
        except Exception as e:
            fail += 1
            print(f"[錯誤] 處理失敗：{in_file} -> {e}", file=sys.stderr)

    return success, fail

# ====================== CLI ======================

def main():
    ap = argparse.ArgumentParser(description="批次清洗 .txt（遞迴）：刪『裝/訂/線』、行首點串、頁碼標籤、行內亂碼。")
    ap.add_argument("input_root", help="輸入根目錄（例如 data）")
    ap.add_argument("output_root", nargs="?", help="輸出根目錄（鏡像結構）。若使用 --inplace 可省略。")
    ap.add_argument("--inplace", action="store_true", help="原地覆寫（小心使用，建議先備份）")
    ap.add_argument("--suffix", default="", help="輸出檔名尾綴（加在副檔名前，如 .clean）")
    ap.add_argument("--quiet", action="store_true", help="安靜模式（不列出每個檔案）")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = None if args.inplace else Path(args.output_root) if args.output_root else None

    try:
        ok, ng = process_tree(
            input_root=input_root,
            output_root=output_root,
            suffix=args.suffix,
            inplace=args.inplace,
            verbose=not args.quiet
        )
        print(f"\n[完成] 成功 {ok} 檔；失敗 {ng} 檔。")
        if not args.inplace and output_root:
            print(f"[輸出路徑] {output_root.resolve()}")
    except Exception as e:
        print(f"[致命錯誤] {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
