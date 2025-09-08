#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import re
import regex as re2
from cleantext import clean
from opencc import OpenCC
import jieba

# ============== 新增：側邊噪音清理 ==============
PUNCT_LINE_RE = re.compile(r'^[\s\.\·⋯…‧・\-—─‐\|／/\\、，,。:：;；!！?？()\[\]【】{}<>〈〉《》"“”\'’`~^＋+＝=\*＊]+$')

SIDEWORDS = {"裝訂線", "騎縫章", "騎縫", "裝訂", "保密", "非公開", "附件", "封條"}

def looks_like_punct_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # 純標點/符號噪音
    if PUNCT_LINE_RE.match(s):
        return True
    # 很短且標點比例高
    non_space = re2.sub(r'\s+', '', s)
    if len(non_space) <= 4:
        punct = sum(1 for ch in non_space if re2.match(r'\p{P}|\p{S}', ch))
        if len(non_space) > 0 and punct / len(non_space) >= 0.6:
            return True
    return False

def remove_margin_artifacts(text: str) -> str:
    """
    移除：
    1) 純標點/符號行
    2) 很短且標點比例高的行
    3) 多行連續單字（縱排）形成的側邊詞，例如「裝 / 訂 / 線」→ 裝訂線
    """
    lines = text.splitlines()

    # 第一輪：丟掉明顯的標點噪音行
    kept = []
    for ln in lines:
        if looks_like_punct_noise(ln):
            continue
        kept.append(ln)

    # 第二輪：偵測連續單字縱排
    res = []
    buf = []

    def flush_buf():
        if not buf:
            return
        joined = ''.join(ch for ch in buf if ch.strip())
        joined_no_punct = re2.sub(r'[\p{P}\p{S}\s]+', '', joined)
        # 偵測是否為常見側邊詞（包含即可）
        drop = any(sw in joined_no_punct for sw in SIDEWORDS)
        if not drop:
            res.extend(buf)
        # 清空
        buf.clear()

    for ln in kept:
        s = ln.strip()
        # 單字行：允許前後有 0~2 個符號（像「. 裝」或「訂 .」）
        if re2.match(r'^[\p{P}\p{S}\s]{0,2}\p{Lo}[\p{P}\p{S}\s]{0,2}$', s):
            # 只保留 Lo（letter other：漢字等）一個字；空白/標點視為噪音
            core = re2.sub(r'[\p{P}\p{S}\s]+', '', s)
            if len(core) == 1:
                buf.append(core)
                continue
        # 遇到非單字行 → 先結算前面的縱排緩衝
        flush_buf()
        res.append(ln)
    flush_buf()

    # 第三輪：壓縮多餘的空白行
    out = []
    prev_blank = False
    for ln in res:
        if ln.strip() == '':
            if not prev_blank:
                out.append('')
            prev_blank = True
        else:
            out.append(ln)
            prev_blank = False

    return '\n'.join(out)

# ============== 第 1 層：clean-text + OpenCC ==============
def stage1_clean_and_convert(text: str, cc_mode: str = "t2tw") -> str:
    t = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_line_breaks=False,
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
    # 去控制/格式字元
    t = re2.sub(r'[\p{Cc}\p{Cf}]', ' ', t)
    # 正規化空白
    t = re2.sub(r'[\p{Zs}\t]+', ' ', t)
    # 清掉多餘的空行空白
    t = re2.sub(r'[ \t]*\n[ \t]*', '\n', t)

    # 先做側邊噪音清除（可放在 OpenCC 前後都行；這裡放前也可）
    t = remove_margin_artifacts(t)

    try:
        cc = OpenCC(cc_mode)  # 's2t', 't2tw', 't2s' 等
        t = cc.convert(t)
    except Exception as e:
        print(f"[警告] OpenCC 初始化失敗（{e}），跳過簡繁轉換。", file=sys.stderr)

    return t.strip()

# ============== 第 2 層：合併錯誤換行 + 斷句規則 ==============
SENT_END = r'[。！？!?；;]'
QUOTE_CLOSERS = '」』”’›》〉）】'
SECTION_HEAD_PAT = re.compile(
    r'^('
    r'(主旨|說明|附件|備註|結論|參考|辦法|依據)\s*[:：]'
    r'|[（(]?[一二三四五六七八九十]\)'
    r'|[一二三四五六七八九十]+、'
    r'|\d+、'
    r')'
)

def merge_broken_lines(t: str) -> str:
    lines = t.splitlines()
    merged = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            merged.append('')
            continue
        next_line = lines[i+1].strip() if i + 1 < len(lines) else ''
        is_section_head = bool(SECTION_HEAD_PAT.match(next_line))
        ends_with_punct = bool(re.search(SENT_END + r'[' + re.escape(QUOTE_CLOSERS) + r']*$', line))
        if not ends_with_punct and not is_section_head:
            if merged:
                merged[-1] = (merged[-1].rstrip() + ' ' + line.lstrip()).strip()
            else:
                merged.append(line)
        else:
            merged.append(line)
    return '\n'.join(merged)

def normalize_punct(t: str) -> str:
    t = re.sub(r'[.]{2,}', '。', t)
    t = re.sub(r'。{2,}', '。', t)
    t = re.sub(r'[，,]{2,}', '，', t)
    t = re.sub(r'[:]{1}', '：', t)
    t = re.sub(r'^\s*[,，。：;；]+', '', t, flags=re.MULTILINE)
    t = t.replace('(', '（').replace(')', '）')
    t = t.replace('[', '【').replace(']', '】')
    t = t.replace('"', '”').replace("'", "’")
    return t

def sentence_split_zh(t: str) -> list:
    t = re2.sub(r'[ \t]+', ' ', t)
    paras = [p.strip() for p in t.split('\n') if p.strip()]
    sents = []
    for p in paras:
        pieces = re2.split(rf'({SENT_END}[{re2.escape(QUOTE_CLOSERS)}]*)(?=$| )', p)
        for i in range(0, len(pieces), 2):
            chunk = pieces[i].strip()
            end = pieces[i+1] if i+1 < len(pieces) else ''
            if not chunk:
                continue
            if end:
                sents.append((chunk + end).strip())
            else:
                if len(chunk) >= 25:
                    sents.append(chunk + '。')
                else:
                    sents.append(chunk)
    sents = [s.strip() for s in sents if re2.sub(r'\p{P}+', '', s).strip()]
    return sents

def basic_punct_fix(sents):
    fixed = []
    for s in sents:
        if SECTION_HEAD_PAT.match(s):
            fixed.append(s if s.endswith('：') else s)
            continue
        if not re.search(r'[。！？；!?]$', s):
            s = s + '。'
        fixed.append(s)
    return fixed

# ============== 主流程 ==============
def process_text(raw: str, cc_mode: str):
    t1 = stage1_clean_and_convert(raw, cc_mode=cc_mode)
    t2 = merge_broken_lines(t1)
    t2 = normalize_punct(t2)
    sents = sentence_split_zh(t2)
    sents2 = basic_punct_fix(sents)
    out = "\n".join(sents2)
    out = normalize_punct(out)
    return out.strip()

def main():
    ap = argparse.ArgumentParser(description="OCR .txt 清洗（含側邊噪音處理；無 LanguageTool）")
    ap.add_argument("input", help="輸入 .txt 檔路徑")
    ap.add_argument("-o", "--output", help="輸出檔名（預設：同名 *.clean.txt）")
    ap.add_argument("--cc", default="t2tw", help="OpenCC 模式（預設 t2tw；常見：s2t, t2tw, t2s）")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[錯誤] 找不到檔案：{in_path}", file=sys.stderr)
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".clean.txt")

    raw = in_path.read_text(encoding="utf-8", errors="ignore")

    print("[INFO] 側邊噪音處理 + 第一層：clean-text + OpenCC…")
    print("[INFO] 第二層：合併錯誤換行 + 中文斷句…")
    print("[INFO] 第三層：規則式補標點（無 LanguageTool）…")
    cleaned = process_text(raw, cc_mode=args.cc)

    out_path.write_text(cleaned, encoding="utf-8")
    print(f"[完成] 已輸出：{out_path}")

if __name__ == "__main__":
    main()
