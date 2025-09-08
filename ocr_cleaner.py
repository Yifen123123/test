#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_cleaner.py — 安全保守的 OCR .txt 清洗器（含你的預設規則 & 強化版條列/點梯處理）

重點：
- 保留欄位（發文日期、發文字號、主旨、說明、XXX 函…）
- 移除雜訊點行、單獨一字（裝/訂/線）、頁碼「第 1 頁，共 2 頁」多變體
- 清除行首的點梯（如「.. 主旨」→「主旨」）、刪除行內三連以上點梯
- 不移除你要保留的亂碼（例如：＄！...）

使用方式：
1) 單檔清洗：python ocr_cleaner.py input.txt
2) 指定輸出：python ocr_cleaner.py input.txt -o cleaned.txt
3) 資料夾遞迴：python ocr_cleaner.py data/raw/通知函 -r
4) 預覽差異：python ocr_cleaner.py input.txt --dry-run
5) 調整強度：--level light|medium|aggressive
6) 追加白/黑名單：--keep 來文字號 --ban "^-{5,}$"
"""
from __future__ import annotations
import argparse
import sys
import unicodedata
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict

# ========== 數字型 token 的誤辨修正 ==========
def _digitish(s: str) -> bool:
    t = re.sub(r"[,./\\\-:：﹕，、\s]", "", s)
    return t != "" and all(ch.isdigit() or ch.isascii() for ch in t)

def fix_numeric_confusions(token: str) -> str:
    if not token:
        return token
    if _digitish(token):
        token = (token
            .replace("O","0").replace("I","1").replace("l","1")
            .replace("Z","2").replace("S","5").replace("B","8"))
    return token

# ========== 白名單：遇到就保留 ==========
DEFAULT_KEEP_KEYWORDS = [
    # 你特別要求的
    "函", "主旨", "電話", "聯絡電話", "聯絡方式", "附件", "地址", "承辦人", "傳真",
    "發文日期", "發文字號", "速別", "說明", "正本", "副本",
    # 常見公文欄位補強
    "受文者", "檔號", "保存年限", "保存期限", "受文單位", "受文機關", "案號", "字號",
    "發文機關", "承辦單位", "核稿", "會簽", "簽發", "來文", "來文字號", "收文日期",
    "機關地址", "機關電話", "頁次",
    # 常見中文數序（條列）
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
]

# ========== 黑名單：但白名單優先 ==========
DEFAULT_BANNED_PATTERNS = [
    r"^\s*$",                                 # 空行
    r"^[\.\·\•\*]{2,}\s*$",                   # 多顆點/圓點/星號的雜訊行
    r"^[\.\s]+$",                             # 純點與空白
    r"^\s*(裝|訂|線)\s*$",                    # 單獨一字（裝/訂/線）行
    r"^第\s*\d+\s*頁\s*[，,、]?\s*共\s*\d+\s*頁\s*$",  # 頁碼行（更寬鬆）
    r"^[\-_]{3,}\s*$",                        # 分隔線
]

LEVELS = {"light": 0, "medium": 1, "aggressive": 2}

# ========== 基本正規化 ==========
def normalize_fullwidth(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)         # 全形→半形（不影響中文）
    s = re.sub(r"[ \t\u3000]+", " ", s)          # 空白歸一
    return s.strip()

def numeric_aware_fix(line: str) -> str:
    parts = re.split(r"(\s+)", line)
    parts = [fix_numeric_confusions(p) if not p.isspace() else p for p in parts]
    return "".join(parts)

def join_hard_wrapped_lines(lines: List[str]) -> List[str]:
    out, buf = [], ""
    for line in lines:
        cur = line.rstrip()
        if buf:
            if (not re.search(r"[。！？：；、,:;)\]\}＞>」』]$", buf)
                and not re.match(r"^([\-–—•\*•]|第[一二三四五六七八九十]條|\d+[\.\)]|\(一\))", cur)):
                buf = buf + " " + cur.lstrip()
            else:
                out.append(buf); buf = cur
        else:
            buf = cur
    if buf:
        out.append(buf)
    return out

def normalize_colons(s: str) -> str:
    return re.sub(r"[：﹕:]\s*", "：", s)

def remove_dot_noise_inline(s: str) -> str:
    # 行首點梯（含空白）→ 去掉（例："..  主旨" → "主旨"）
    s = re.sub(r'^(?:[\.\s]{1,})', '', s)
    # 行內三連以上點梯 → 視為版面雜訊，移除並以單一空白替代
    s = re.sub(r'(?:\s*\.\s*){3,}', ' ', s)
    # 行尾兩連以上點梯 → 移除
    s = re.sub(r'(?:\s*\.\s*){2,}$', '', s)
    return s

def normalize_bullets(s: str) -> str:
    # 條列符號與括號編號正規化
    s = re.sub(r'^[\*\•\·]\s*', '・', s)                 # • * · → ・
    s = re.sub(r'^(\d+)[\)\.]\s?', r'\1. ', s)           # 1) / 1. → 1. 
    s = re.sub(r'^[\(\（](\d+)[\)\）]\s*', r'\1. ', s)   # (1) / （1） → 1.
    s = re.sub(r'^[\(\（](一|二|三|四|五|六|七|八|九|十)[\)\）]\s*', r'\1、', s)  # （一）→ 一、
    return s

def drop_noise_lines(lines: List[str], keep_keywords: List[str], banned_patterns: List[str], level: int) -> List[str]:
    out = []
    bans = [re.compile(p) for p in banned_patterns]
    for line in lines:
        raw = line
        line = line.rstrip()
        if any(kw in line for kw in keep_keywords):
            out.append(raw); continue
        if any(b.search(line) for b in bans):
            continue
        # 極短英文殘塊（medium+），不含中文才刪
        if level >= 1:
            if len(line.strip()) <= 1 and not re.search(r"[\u4e00-\u9fff]", line):
                continue
        out.append(raw)
    return out

def collapse_dot_ladders(s: str) -> str:
    # 合併多段點梯，再交由 remove_dot_noise_inline 做後續清理
    s = re.sub(r"(?:\.\s*){3,}", "...", s)
    s = s.replace("．．．", "…").replace("… …", "…")
    return s

def remove_header_footer(lines: List[str], level: int) -> List[str]:
    if len(lines) < 40:  # 單頁不處理（避免誤殺）
        return lines
    freq: Dict[str, int] = {}
    for ln in lines:
        l = ln.strip()
        if not l: continue
        freq[l] = freq.get(l, 0) + 1
    out = []
    for ln in lines:
        l = ln.strip()
        keep = True
        if 2 <= freq.get(l, 0) and len(l) < 30 and level >= 1:
            if not any(kw in l for kw in DEFAULT_KEEP_KEYWORDS):
                keep = False
        if keep:
            out.append(ln)
    return out

@dataclass
class CleanerConfig:
    encoding: str = "utf-8"
    level: int = LEVELS["medium"]
    keep_keywords: List[str] = field(default_factory=lambda: list(DEFAULT_KEEP_KEYWORDS))
    banned_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_BANNED_PATTERNS))

def clean_text(content: str, cfg: CleanerConfig) -> str:
    # 1) 基本正規化
    content = normalize_fullwidth(content)
    lines = content.splitlines()

    # 2) 版面級處理
    lines = remove_header_footer(lines, cfg.level)
    lines = join_hard_wrapped_lines(lines)

    # 3) 先清掉會擋住白名單偵測的點梯雜訊
    lines = [remove_dot_noise_inline(ln) for ln in lines]

    # 4) 行級刪除（白名單優先）
    lines = drop_noise_lines(lines, cfg.keep_keywords, cfg.banned_patterns, cfg.level)

    # 5) 行內微清洗
    cleaned_lines = []
    for ln in lines:
        s = ln.rstrip("\n")
        s = normalize_colons(s)
        s = collapse_dot_ladders(s)
        s = remove_dot_noise_inline(s)
        s = numeric_aware_fix(s)
        s = normalize_bullets(s)
        cleaned_lines.append(s.strip())

    # 6) 相鄰重複行壓縮（白名單行不壓）
    final_lines, prev = [], None
    for s in cleaned_lines:
        if s and s == prev and not any(kw in s for kw in cfg.keep_keywords):
            continue
        final_lines.append(s); prev = s
    return "\n".join(final_lines).strip() + "\n"

# ========== 檔案 I/O ==========
def read_text(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="ignore") as f:
        return f.read()

def write_text(path: Path, content: str, encoding: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="\n") as f:
        f.write(content)

def process_file(in_path: Path, out_path: Path|None, cfg: CleanerConfig, dry_run: bool=False):
    raw = read_text(in_path, cfg.encoding)
    cleaned = clean_text(raw, cfg)
    if dry_run:
        import difflib
        diff = difflib.unified_diff(
            raw.splitlines(), cleaned.splitlines(),
            fromfile=str(in_path),
            tofile=str(out_path or (in_path.with_suffix('').name + '_clean.txt')),
            lineterm=''
        )
        sys.stdout.write("\n".join(diff) + "\n")
        return in_path, None
    if out_path is None:
        out_path = in_path.with_name(in_path.stem + "_clean.txt")
    write_text(out_path, cleaned, cfg.encoding)
    return in_path, out_path

def walk_and_process(root: Path, cfg: CleanerConfig, dry_run: bool=False):
    for p in root.rglob("*.txt"):
        rel = p.relative_to(root)
        out_p = root.parent / ("cleaned_" + root.name) / rel
        process_file(p, out_p, cfg, dry_run=dry_run)

# ========== CLI ==========
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="保守的 OCR .txt 清洗器（保留欄位、不洗壞）")
    ap.add_argument("path", help=".txt 檔或資料夾")
    ap.add_argument("-o", "--output", default=None, help="輸出檔名（單檔）")
    ap.add_argument("-r", "--recursive", action="store_true", help="資料夾遞迴處理（輸出到 cleaned_<folder>）")
    ap.add_argument("--dry-run", action="store_true", help="僅顯示差異，不輸出")
    ap.add_argument("--level", choices=list(LEVELS.keys()), default="medium", help="清洗強度")
    ap.add_argument("--keep", nargs="*", default=None, help="追加必留關鍵字")
    ap.add_argument("--ban", nargs="*", default=None, help="追加必刪樣式（正則）")
    ap.add_argument("--encoding", default="utf-8", help="讀寫編碼（預設 utf-8，可改 cp950 / big5 等）")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    target = Path(args.path)
    cfg = CleanerConfig(encoding=args.encoding, level=LEVELS[args.level])
    if args.keep: cfg.keep_keywords.extend(_
