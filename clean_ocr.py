# clean_ocr_safe.py
# -*- coding: utf-8 -*-
"""
更保守的 OCR 清洗器：
- 必留白名單（法院/機關抬頭、主旨/說明/依據、檔號/保存年限、發文字號/發文日期、受文者、地址/電話/傳真、案號/字號、日期/字第…號、身分證等）
- 避免把表頭/欄位與前後行黏在一起
- 仍會清掉純噪音行、頁眉頁尾（可選）、縱排殘影、過度標點、軟換行

建議流程：
1) 先 --stdout 做幾檔確認，再批次跑 --in_dir
"""

from __future__ import annotations
import argparse, re, sys, unicodedata
from pathlib import Path
from typing import List

# ---------- 文字正規化 ----------
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"]
NBSP = "\u00a0"

def normalize_unicode(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    for ch in ZERO_WIDTH:
        t = t.replace(ch, "")
    t = t.replace(NBSP, " ")
    return t.replace("\r\n", "\n").replace("\r", "\n")

# ---------- 判斷規則 ----------

# 必留白名單（命中任一就保留）
KEEP_PATTERNS = [
    r"(法院|地方法院|高等法院|最高法院|行政執行署|執行處)",  # 機關/法院抬頭
    r"(主旨|主文|說明|理由|依據)\s*[:：]",                   # 主要欄位
    r"(檔號|檔 案 號|檔案號)\s*[:：]",                       # 檔號
    r"(保存年限|保存期限)\s*[:：]",                           # 保存年限
    r"(受文者|受文機關)\s*[:：]",                             # 受文者
    r"(發文字號|發文日期|來文日期|來文字號|文號)\s*[:：]",     # 文號/日期
    r"(地址|聯絡地址)\s*[:：]",                               # 地址
    r"(電話|聯絡方式|傳真|承辦人)\s*[:：]",                    # 連絡欄
    r"(案號|字號|字第)\s*[:：]?",                              # 案號/字號
    r"(執行命令|扣押命令|撤銷令|收取令|通知函)",               # 文種
    r"\b[0-9]{2,3}[./-][0-9]{1,2}[./-][0-9]{1,2}\b",          # 114.01.02 / 2025-09-05
    r"(?:中華民國)?\s*[0-9]{3,4}\s*年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日",  # 中文日期
    r"(?:司執|家執|保字|金管)\S*第?\s*\d+\s*號",               # 常見字第…號
    r"字第\s*\d+\s*號",
    r"\b[A-Z][12]\d{8}\b",                                    # 身分證
]

KEEP_RE = [re.compile(p) for p in KEEP_PATTERNS]

def must_keep_line(s: str, idx: int, head_keep_lines: int) -> bool:
    """命中白名單或位於前 N 行且像表頭，就保留。"""
    ss = s.strip()
    if not ss:
        return False
    # 前 N 行的寬鬆保留：包含冒號且是中文開頭，當表頭
    if idx < head_keep_lines and re.search(r"[:：]", ss) and re.match(r"^[\u4e00-\u9fff]", ss):
        return True
    for r in KEEP_RE:
        if r.search(ss):
            return True
    return False

# 純噪音行（只有這堆符號）
NOISE_CHARS = r"\.\·•・‧∙⋯…,_\-—–=~\*\+|<>〈〉《》【】〔〕［］\[\]{}（）\(\)○、，。；：:?!？！／/\\‧‥^`'\" "
NOISE_LINE_RE = re.compile(rf"^[{NOISE_CHARS}]+$")

def is_noise_line(s: str) -> bool:
    ss = s.strip()
    if not ss:
        return False  # 空行保留，最後再收斂
    if NOISE_LINE_RE.match(ss):
        return True
    # 極短且標點比例極高（<=3字，且去標點後≤1字）
    core = re.sub(rf"[{NOISE_CHARS}]", "", ss)
    if len(ss) <= 3 and len(core) <= 1:
        return True
    return False

# 頁碼/頁眉（可選保留）
PAGE_RE_LIST = [
    re.compile(r"^第\s*\d+\s*頁(?:\s*/\s*共\s*\d+\s*頁)?\s*$"),
    re.compile(r"^Page\s*\d+(?:\s*/\s*\d+)?\s*$", re.I),
    re.compile(r"^\s*-+\s*\d+\s*-+\s*$"),
    re.compile(r"^\s*_{2,}\s*$"),
]

def is_page_line(s: str) -> bool:
    ss = s.strip()
    for r in PAGE_RE_LIST:
        if r.match(ss):
            return True
    return False

# 條列/段首（避免被併行）
PARA_START_RE = re.compile(
    r"^\s*(?:"
    r"[（(]?[一二三四五六七八九十][)）、\.．]"      # （一）、一.
    r"|[0-9]{1,2}[)\.、．]"                       # 1)、2.
    r"|[一二三四五六七八九十]+、"                  # 一、二、
    r"|主旨|主文|說明|理由|依據"                   # 主要欄位
    r"|檔號|保存年限|受文者|發文字號|發文日期|地址|電話|傳真|案號|字號"  # 表頭欄位
    r")\s*[:：]?"
)

# 句末符號（用來判斷是否合併軟換行）
END_PUNC = "。！？；：…】）〉》』」"
SOFT_WRAP_END_OK_RE = re.compile(rf"[{END_PUNC}\s]$")

# 英文行末連字號
HYPHEN_LINE_RE = re.compile(r"[A-Za-z]\-$")

# 縱排殘影（連續很多 1~2 字的短行）
VERT_SHORT_LEN = 2

def merge_vertical_blocks(lines: List[str], min_run: int = 6, drop_words: set[str] | None = None) -> List[str]:
    """把連續很多 1~2字的行合併；若合成「裝訂線/撕裂線」等詞就整段丟掉。"""
    drop_words = drop_words or {"裝訂線", "撕裂線", "裝訂處"}
    out: List[str] = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and 0 < len(lines[j].strip()) <= VERT_SHORT_LEN:
            j += 1
        run = j - i
        if run >= min_run:
            merged = "".join([lines[k].strip() for k in range(i, j)])
            if 1 <= len(merged) <= 6 and re.fullmatch(r"[\u4e00-\u9fff]+", merged) and (merged in drop_words):
                pass  # 整段丟
            else:
                out.append(merged)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out

def collapse_runs(s: str) -> str:
    # 收斂連續標點與空白，但保留內容
    s = re.sub(r"[\.⋯…]{2,}", "……", s)  # 六點
    s = re.sub(r"[—–\-]{2,}", "—", s)
    s = re.sub(r"\*{2,}", "*", s)
    s = re.sub(r"_{2,}", "—", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s

def merge_soft_wraps(lines: List[str]) -> List[str]:
    """合併軟換行：上一行不是句末，且下一行不是段首/表頭"""
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            # 英文 hyphenation
            if HYPHEN_LINE_RE.search(cur.strip()):
                cur = re.sub(r"-$", "", cur.rstrip()) + nxt.lstrip()
                i += 2
                lines[i-1] = cur
                continue
            # 段首與表頭不可合併
            if not SOFT_WRAP_END_OK_RE.search(cur.strip()) and not PARA_START_RE.search(nxt):
                joiner = "" if re.search(r"[\u4e00-\u9fff]$", cur) and re.search(r"^[\u4e00-\u9fff]", nxt) else " "
                cur = cur.rstrip() + joiner + nxt.lstrip()
                i += 2
                lines[i-1] = cur
                continue
        out.append(cur)
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

# ---------- 主清洗流程 ----------

def clean_text(text: str,
               head_keep_lines: int = 30,
               vertical_min_run: int = 6,
               drop_page_lines: bool = True,
               max_blank_lines: int = 1) -> str:
    # 1) 正規化
    t = normalize_unicode(text)
    raw_lines = t.split("\n")

    # 2) 逐行篩選：頁碼/噪音，但先看是否「必留」
    kept: List[str] = []
    for idx, s in enumerate(raw_lines):
        ss = s.rstrip()
        if must_keep_line(ss, idx, head_keep_lines):
            kept.append(ss)
            continue
        if drop_page_lines and is_page_line(ss):
            continue
        if is_noise_line(ss):
            continue
        kept.append(ss)

    if not kept:
        return ""

    # 3) 縱排殘影（保守）
    kept = merge_vertical_blocks(kept, min_run=vertical_min_run)

    # 4) 合併軟換行（可重複）
    prev_len = None
    while prev_len != len(kept):
        prev_len = len(kept)
        kept = merge_soft_wraps(kept)

    # 5) 收斂標點/空白
    kept = [collapse_runs(s) for s in kept]

    # 6) 收斂空白行
    kept = collapse_blank_lines(kept, max_blank=max_blank_lines)

    return "\n".join(kept).strip("\n \t\r")

# ---------- 入口 ----------

def process_file(in_path: Path, out_path: Path,
                 head_keep_lines: int,
                 vertical_min_run: int,
                 drop_page_lines: bool,
                 max_blank_lines: int) -> None:
    try:
        text = in_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[ERR] 讀檔失敗：{in_path} → {e}", file=sys.stderr)
        return
    cleaned = clean_text(
        text=text,
        head_keep_lines=head_keep_lines,
        vertical_min_run=vertical_min_run,
        drop_page_lines=not args.keep_page_lines if 'args' in globals() else drop_page_lines,
        max_blank_lines=max_blank_lines,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")

def walk_and_process(in_dir: Path, out_dir: Path,
                     head_keep_lines: int,
                     vertical_min_run: int,
                     drop_page_lines: bool,
                     max_blank_lines: int,
                     overwrite: bool = True,
                     pattern: str = "*.txt") -> None:
    files = sorted(in_dir.rglob(pattern))
    n = 0
    for p in files:
        rel = p.relative_to(in_dir)
        out_p = out_dir / rel
        if not overwrite and out_p.exists():
            continue
        process_file(p, out_p, head_keep_lines, vertical_min_run, drop_page_lines, max_blank_lines)
        n += 1
    print(f"[OK] 已清洗 {n} 個檔案 → {out_dir}")

def main():
    global args
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in_dir", type=str, help="輸入資料夾（遞迴處理 .txt）")
    g.add_argument("--in_file", type=str, help="單一 .txt 檔案")
    ap.add_argument("--out_dir", type=str, default="cleaned", help="輸出資料夾")
    ap.add_argument("--stdout", action="store_true", help="單檔模式下，輸出到 stdout 而不是寫檔")
    ap.add_argument("--pattern", type=str, default="*.txt", help="遞迴模式的檔名匹配")
    # 重要可調參數（都偏保守）
    ap.add_argument("--head_keep_lines", type=int, default=30, help="前 N 行視為表頭區，含冒號的中文行會保留（預設 30 行）")
    ap.add_argument("--vertical_min_run", type=int, default=6, help="縱排殘影合併的最小連續短行數（預設 6）")
    ap.add_argument("--keep_page_lines", action="store_true", help="保留頁碼/頁眉（預設移除）")
    ap.add_argument("--max_blank_lines", type=int, default=1, help="最多保留的連續空白行數")
    ap.add_argument("--no_overwrite", action="store_true", help="已存在就不覆蓋")
    args = ap.parse_args()

    if args.in_file:
        p = Path(args.in_file)
        txt = p.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(
            txt,
            head_keep_lines=args.head_keep_lines,
            vertical_min_run=args.vertical_min_run,
            drop_page_lines=not args.keep_page_lines,
            max_blank_lines=args.max_blank_lines,
        )
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
        head_keep_lines=args.head_keep_lines,
        vertical_min_run=args.vertical_min_run,
        drop_page_lines=not args.keep_page_lines,
        max_blank_lines=args.max_blank_lines,
        overwrite=not args.no_overwrite,
        pattern=args.pattern
    )

if __name__ == "__main__":
    main()
