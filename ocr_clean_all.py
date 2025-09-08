#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR .txt 清洗流程（無 LanguageTool，含刪除「裝/訂/定/線」與雜訊）：
1) 黑名單字元移除（裝、訂、定、線 + 简体）
2) clean-text 去除控制字元/亂碼與基礎正規化
3) OpenCC 簡繁統一（預設：t2tw，簡→正體台灣慣用字）
4) 合併錯誤換行、統一標點、中文斷句
5) 規則式補句點、去除純標點/符號行（例如 .. . .. ...）

依賴：
pip install "clean-text[gpl]" opencc-python-reimplemented jieba regex
"""

import argparse
from pathlib import Path
import sys
import re
import regex as re2  # 支援 \p{...}
from cleantext import clean
from opencc import OpenCC
import jieba

# -------------------- 參數：字元黑名單（可透過 --blacklist 覆蓋/擴充） --------------------
DEFAULT_BLACKLIST = "裝订訂装定線线"  # 含繁/簡 + 「定」

def remove_blacklist_chars(text: str, blacklist: str) -> str:
    """逐字過濾指定黑名單字元；若行因此變空則一併清除。"""
    bl_set = set(list(blacklist))
    # 逐字過濾
    text = "".join(ch for ch in text if ch not in bl_set)
    # 刪掉空行
    lines = text.splitlines()
    lines = [ln for ln in lines if ln.strip() != ""]
    return "\n".join(lines)

# -------------------- 純標點/符號行 的過濾 --------------------
PUNCT_ONLY_RE = re2.compile(r'^\s*[\p{P}\p{S}]+\s*$')

def drop_punct_only_lines(text: str) -> str:
    lines = text.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # 純標點/符號（含 .. ... ／ ｜ 等） -> 丟掉
        if PUNCT_ONLY_RE.match(s):
            continue
        kept.append(ln)
    return "\n".join(kept)

# -------------------- 第 1 層：clean-text + OpenCC --------------------
def stage1_clean_and_convert(text: str, cc_mode: str = "t2tw") -> str:
    """清亂碼/控制字元/空白，並做簡繁轉換"""
    t = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_line_breaks=False,     # 保留換行，第二層再處理
        no_urls=False,
        no_emails=False,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="。",
        replace_with_url=" ",
        replace_with_email=" ",
        replace_with_phone_number=" ",
        replace_with_number=" ",
        replace_with_digit=" ",
        replace_with_currency_symbol=" ",
    )
    # 去除控制/格式字元（如 \x0c, \u200b 等）
    t = re2.sub(r'[\p{Cc}\p{Cf}]', ' ', t)
    # 正規化空白（含全形空白）
    t = re2.sub(r'[\p{Zs}\t]+', ' ', t)
    # 清掉多餘的空行空白
    t = re2.sub(r'[ \t]*\n[ \t]*', '\n', t)

    try:
        cc = OpenCC(cc_mode)  # 常用：'s2t', 't2tw', 't2s'
        t = cc.convert(t)
    except Exception as e:
        print(f"[警告] OpenCC 初始化失敗（{e}），跳過簡繁轉換。", file=sys.stderr)

    return t.strip()

# -------------------- 第 2 層：合併錯誤換行 + 斷句規則 --------------------
SENT_END = r'[。！？!?；;]'
QUOTE_CLOSERS = '」』”’›》〉）】'  # 修正：移除多餘的 ']'
QUOTE_CLOSERS_RE = re.escape(QUOTE_CLOSERS)

SECTION_HEAD_PAT = re.compile(
    r'^('
    r'(主旨|說明|附件|備註|結論|參考|辦法|依據)\s*[:：]'
    r'|[（(]?[一二三四五六七八九十]\)'
    r'|[一二三四五六七八九十]+、'
    r'|\d+、'
    r')'
)

def merge_broken_lines(t: str) -> str:
    """
    合併不應該換行的行：
    - 若行尾無句末標點，且下一行不是段落標頭，就把換行替換成空白。
    - 對「主旨：／說明：」等段首，保留換行。
    """
    lines = t.splitlines()
    merged = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            merged.append('')
            continue

        next_line = lines[i+1].strip() if i + 1 < len(lines) else ''
        is_section_head = bool(SECTION_HEAD_PAT.match(next_line))
        ends_with_punct = bool(re.search(SENT_END + r'[' + QUOTE_CLOSERS_RE + r']*$', line))

        if not ends_with_punct and not is_section_head:
            # 行尾無句末標點、下一行不是段首 => 連成同一句
            if merged:
                merged[-1] = (merged[-1].rstrip() + ' ' + line.lstrip()).strip()
            else:
                merged.append(line)
        else:
            merged.append(line)
    return '\n'.join(merged)

def normalize_punct(t: str) -> str:
    """統一常見標點、收斂連續標點"""
    t = re.sub(r'[.]{2,}', '。', t)      # 英文句號連發 → 中文句號
    t = re.sub(r'。{2,}', '。', t)        # 多個中文句號 → 一個
    t = re.sub(r'[，,]{2,}', '，', t)     # 逗號連發 → 一個
    t = re.sub(r'[:]{1}', '：', t)        # 冒號 → 中文冒號
    t = re.sub(r'^\s*[,，。：;；]+', '', t, flags=re.MULTILINE)  # 行首多餘標點
    # 括號/引號中英文統一（保守處理，只替換常見半形）
    t = t.replace('(', '（').replace(')', '）')
    t = t.replace('[', '【').replace(']', '】')
    t = t.replace('"', '”')               # 粗略處理；若要更精細可擴充
    t = t.replace("'", "’")
    return t

def sentence_split_zh(t: str) -> list:
    """
    以中文標點為主進行斷句，保留引號在句末。
    對少標點長行，使用長度閾值自動補句號。
    """
    t = re2.sub(r'[ \t]+', ' ', t)  # 段內多餘空白縮成單一空白
    paras = [p.strip() for p in t.split('\n') if p.strip()]
    sents = []
    for p in paras:
        # 以句末標點 + 可能結尾引號切分
        pieces = re2.split(rf'({SENT_END}[{re2.escape(QUOTE_CLOSERS)}]*)(?=$| )', p)
        for i in range(0, len(pieces), 2):
            chunk = pieces[i].strip()
            end = pieces[i+1] if i+1 < len(pieces) else ''
            if not chunk:
                continue
            if end:
                sents.append((chunk + end).strip())
            else:
                # 無句末標點的片段：若過長，補句號
                if len(chunk) >= 25:
                    sents.append(chunk + '。')
                else:
                    sents.append(chunk)
    # 去除過短且無資訊的碎片（只剩標點或空白）
    sents = [s.strip() for s in sents if re2.sub(r'\p{P}+', '', s).strip()]
    return sents

def basic_punct_fix(sents):
    """
    規則式補標點（改良版 v2）：
    - 不補在段首標題（主旨：/說明：…）
    - 不補在冒號、括號、右引號結尾
    - 不補在包含「身分證」「統一編號」的句子
    - 不補在以「號/号」結尾的句子
    - 其餘才補句號
    """
    fixed = []
    for s in sents:
        s = s.strip()

        # 段首標題直接保留
        if SECTION_HEAD_PAT.match(s):
            fixed.append(s if s.endswith('：') else s)
            continue

        # 不補句號的情境
        if s.endswith(("：", "）", "】", "」", "』")):
            fixed.append(s)
            continue
        if "身分證" in s or "統一編號" in s:
            fixed.append(s)
            continue
        if re.search(r"(號|号)$", s):
            fixed.append(s)
            continue

        # 已以標點結尾就保留
        if re.search(r'[。！？；!?]$', s):
            fixed.append(s)
        else:
            fixed.append(s + "。")
    return fixed

# -------------------- 主流程 --------------------
def process_text(raw: str, cc_mode: str, blacklist: str):
    # 0) 先移除黑名單字元（裝/訂/定/線 + 简体）
    t0 = remove_blacklist_chars(raw, blacklist=blacklist)
    # 0.5) 移除純標點/符號行（如 .. . .. ...）
    t0 = drop_punct_only_lines(t0)

    # 1) 清洗 + 簡繁
    t1 = stage1_clean_and_convert(t0, cc_mode=cc_mode)

    # 2) 合併換行 + 標點統一 + 斷句
    t2 = merge_broken_lines(t1)
    t2 = normalize_punct(t2)
    sents = sentence_split_zh(t2)

    # 3) 規則式補標點
    sents2 = basic_punct_fix(sents)

    # 4) 一行一句輸出
    out = "\n".join(sents2)
    out = normalize_punct(out)
    return out.strip()

def main():
    ap = argparse.ArgumentParser(description="OCR .txt 清洗（刪除『裝/訂/定/線』與雜訊；無 LanguageTool 版）")
    ap.add_argument("input", help="輸入 .txt 檔路徑")
    ap.add_argument("-o", "--output", help="輸出檔名（預設：同名 *.clean.txt）")
    ap.add_argument("--cc", default="t2tw", help="OpenCC 模式（預設 t2tw；常見：s2t, t2tw, t2s）")
    ap.add_argument("--blacklist", default=DEFAULT_BLACKLIST, help=f"要刪除的字元集合（預設：{DEFAULT_BLACKLIST}）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[錯誤] 找不到檔案：{in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".clean.txt")

    # 讀取（容忍編碼錯誤）
    raw = in_path.read_text(encoding="utf-8", errors="ignore")

    print("[INFO] 0) 黑名單字元與純標點行處理…")
    print("[INFO] 1) clean-text + OpenCC…")
    print("[INFO] 2) 合併錯誤換行 + 中文斷句…")
    print("[INFO] 3) 規則式補標點（無 LanguageTool）…")

    cleaned = process_text(raw, cc_mode=args.cc, blacklist=args.blacklist)

    out_path.write_text(cleaned, encoding="utf-8")
    print(f"[完成] 已輸出：{out_path}")

if __name__ == "__main__":
    main()
