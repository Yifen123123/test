# clean_ocr.py
# -*- coding: utf-8 -*-
"""
清洗 OCR 的 .txt：
- Unicode 正規化、清掉零寬空白/BOM
- 刪除純噪音行（只有 . … — * 等裝飾符）
- 移除頁首頁尾（第 X 頁、Page X / X、—— 1 ——）
- 合併軟換行（句中被硬切行）
- 修正英文字 hyphenation（行末 - 接下一行）
- 合併「縱排殘影」短行（連續很多長度≤2 的中文行）
- 收斂連續標點（......、———、＊＊＊）
- 可刪除常見頁邊提示（如「裝訂線」「撕裂線」）
"""

from __future__ import annotations
import argparse, re, sys, unicodedata
from pathlib import Path
from typing import List

# ---------- 基本規則 ----------
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
NBSP = "\u00a0"

# 「純噪音行」：只有這些符號就刪掉
NOISE_CHARS = r"\.\·•・‧∙⋯…,_\-—–=~\*\+|<>〈〉《》【】〔〕［］\[\]{}（）\(\)○、，。；：:?!？！／/\\‧‥^`'\" "
NOISE_LINE_RE = re.compile(rf"^[{NOISE_CHARS}]+$")

# 頁碼/頁首頁尾
PAGE_RE_LIST = [
    re.compile(r"^第\s*\d+\s*頁(?:\s*/\s*共\s*\d+\s*頁)?\s*$"),
    re.compile(r"^Page\s*\d+(?:\s*/\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*-+\s*\d+\s*-+\s*$"),
    re.compile(r"^\s*_{2,}\s*$"),
]

# 清單/段落開頭（不要合併）
PARA_START_RE = re.compile(
    r"^\s*(?:[（(]?[一二三四五六七八九十][)）、\.．]|[0-9]{1,2}[)\.、．]|[一二三四五六七八九十]+、|- |• |‧ |． )"
)

# 合併條件：行尾是否像句末
END_PUNC = "。！？；：…】）〉》』」"
SOFT_WRAP_END_OK_RE = re.compile(rf"[{END_PUNC}\s]$")

# 英文單字 + 行末連字號（hyphenation）
HYPHEN_LINE_RE = re.compile(r"[A-Za-z]\-$")

# 縱排殘影：連續 N 行，每行長度 ≤ M
VERT_SHORT_LEN = 2
VERT_MIN_RUN = 5

# 連續符號收斂
ELLIPSIS_RE = re.compile(r"[\.⋯…]{2,}")
DASH_RE = re.compile(r"[—–\-]{2,}")
STAR_RE = re.compile(r"\*{2,}")
UNDER_RE = re.compile(r"_{2,}")

# 常見頁邊提示（遇到由縱排拼出的短詞時可整句丟掉）
DROP_WORDS = {"裝訂線", "撕裂線", "裝訂處"}

def normalize_unicode(text: str) -> str:
    # 轉 NFKC，去 BOM/零寬空白，NBSP -> 普通空白
    t = unicodedata.normalize("NFKC", text)
    for ch in ZERO_WIDTH:
        t = t.replace(ch, "")
    t = t.replace(NBSP, " ")
    # 統一換行
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    return t

def is_noise_line(s: str) -> bool:
    # 純噪音行或空白
    if not s.strip():
        return False  # 空行保留，稍後收斂
    if NOISE_LINE_RE.match(s):
        return True
    # 過高標點比例（>70%）且長度≤6，也視為噪音
    pure = re.sub(rf"[{NOISE_CHARS}]", "", s)
    if len(s.strip()) <= 6 and len(pure) <= 2:
        return True
    return False

def is_page_line(s: str) -> bool:
    ss = s.strip()
    for r in PAGE_RE_LIST:
        if r.match(ss):
            return True
    return False

def collapse_runs(s: str) -> str:
    s = ELLIPSIS_RE.sub("……", s)   # 中文六點
    s = DASH_RE.sub("—", s)
    s = STAR_RE.sub("*", s)
    s = UNDER_RE.sub("—", s)        # 下劃線轉長橫
    # 多個空白 → 一個
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def merge_soft_wraps(lines: List[str]) -> List[str]:
    """合併軟換行：上一行不是句末，且下一行不是條列起頭/段落開頭"""
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            # hyphenation：英文字 + 行末連字號
            if HYPHEN_LINE_RE.search(cur.strip()):
                cur = re.sub(r"-$", "", cur.rstrip()) + nxt.lstrip()
                i += 2
                lines[i-1] = cur
                continue
            # 若當前行不是句末，下一行也不是段落起頭 → 併行
            if not SOFT_WRAP_END_OK_RE.search(cur.strip()) and not PARA_START_RE.search(nxt):
                # 中文之間直接連，英中混合加單空格
                joiner = "" if re.search(r"[\u4e00-\u9fff]$", cur) and re.search(r"^[\u4e00-\u9fff]", nxt) else " "
                cur = cur.rstrip() + joiner + nxt.lstrip()
                i += 2
                # 併完以 cur 取代下一行，繼續嘗試再併
                lines[i-1] = cur
                continue
        out.append(cur)
        i += 1
    return out

def merge_vertical_blocks(lines: List[str]) -> List[str]:
    """把連續很多 1~2 字的行合併（縱排殘影）；若合出來是“裝訂線”等短詞就丟掉"""
    out: List[str] = []
    i = 0
    while i < len(lines):
        # 找連續短行
        j = i
        while j < len(lines) and len(lines[j].strip()) <= VERT_SHORT_LEN and lines[j].strip():
            j += 1
        run_len = j - i
        if run_len >= VERT_MIN_RUN:
            merged = "".join([lines[k].strip() for k in range(i, j)])
            if (len(merged) <= 6 and re.fullmatch(r"[\u4e00-\u9fff]+", merged)) and (merged in DROP_WORDS):
                # 直接丟掉整段
                pass
            else:
                out.append(merged)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out

def collapse_blank_lines(lines: List[str], max_blank: int = 1) -> List[str]:
    out: List[str] = []
    blanks = 0
    for s in lines:
        if s.strip():
            blanks = 0
            out.append(s)
        else:
            blanks += 1
            if blanks <= max_blank:
                out.append("")
    return out

def clean_text(text: str,
               max_blank_lines: int = 1,
               keep_ellipsis: int = 6) -> str:
    # 1) 正規化
    t = normalize_unicode(text)

    # 2) 分行 → 移除噪音行/頁碼行
    raw_lines = t.split("\n")
    lines = []
    for s in raw_lines:
        ss = s.rstrip()
        if is_page_line(ss):
            continue
        if is_noise_line(ss):
            continue
        lines.append(ss)

    if not lines:
        return ""

    # 3) 合併縱排殘影
    lines = merge_vertical_blocks(lines)

    # 4) 合併軟換行（可重複數次直到不再變短）
    prev = None
    while prev != len(lines):
        prev = len(lines)
        lines = merge_soft_wraps(lines)

    # 5) 收斂連續標點
    lines = [collapse_runs(s) for s in lines]

    # 6) 收斂空白行
    lines = collapse_blank_lines(lines, max_blank=max_blank_lines)

    # 7) 結尾修飾
    out = "\n".join(lines).strip("\n \t\r")
    # 若想把 …… 換成 3 個點，可在此調整
    if keep_ellipsis not in (6, 3):
        # 其他長度：按需求處理
        pass
    return out

def process_file(in_path: Path, out_path: Path,
                 max_blank_lines: int = 1) -> None:
    try:
        text = in_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[ERR] 讀檔失敗：{in_path} → {e}", file=sys.stderr)
        return
    cleaned = clean_text(text, max_blank_lines=max_blank_lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")

def walk_and_process(in_dir: Path, out_dir: Path,
                     max_blank_lines: int = 1,
                     overwrite: bool = True,
                     pattern: str = "*.txt") -> None:
    files = sorted(in_dir.rglob(pattern))
    n = 0
    for p in files:
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel
        if not overwrite and out_p.exists():
            continue
        process_file(p, out_p, max_blank_lines=max_blank_lines)
        n += 1
    print(f"[OK] 已清洗 {n} 個檔案 → {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in_dir", type=str, help="輸入資料夾（遞迴處理 .txt）")
    g.add_argument("--in_file", type=str, help="單一 .txt 檔案")
    ap.add_argument("--out_dir", type=str, default="cleaned", help="輸出資料夾")
    ap.add_argument("--stdout", action="store_true", help="單檔模式下，輸出到 stdout 而不是寫檔")
    ap.add_argument("--pattern", type=str, default="*.txt", help="遞迴模式的檔名匹配")
    ap.add_argument("--max_blank_lines", type=int, default=1, help="最多保留的連續空白行數")
    ap.add_argument("--no_overwrite", action="store_true", help="已存在就不覆蓋")
    args = ap.parse_args()

    if args.in_file:
        p = Path(args.in_file)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(txt, max_blank_lines=args.max_blank_lines)
        if args.stdout:
            sys.stdout.write(cleaned)
        else:
            out_p = Path(args.out_dir) / p.name
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(cleaned, encoding="utf-8")
            print(f"[OK] {p} → {out_p}")
        return

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    walk_and_process(
        in_dir, out_dir,
        max_blank_lines=args.max_blank_lines,
        overwrite=not args.no_overwrite,
        pattern=args.pattern
    )

if __name__ == "__main__":
    main()
