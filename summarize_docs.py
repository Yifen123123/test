# summarize_docs.py
# -*- coding: utf-8 -*-
"""
產出公文的結構化摘要與短文摘要（for 外勤快速消化）。
- 從 records_csv 取得 doc_id / label（或 pred_csv 的 pred）
- 從 raw/<label>/<doc_id>.txt 讀全文（或指定 --fulltext_col 直接用欄位）
- 抽取：機關、案號、依據、角色姓名、人證、保單號、動作、期限/日內、急件
- 產生：summary_text（行動導向）、summaries.jsonl 與 summaries.csv

修正點：避免把「債務人：名下所有 / 清償」當作姓名。
"""

import re
import json
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ----------------------------
# 常用詞 / 正則
# ----------------------------

ROLE_TOKENS = ["要保人","被保險人","受益人","債務人","債權人","公職人員","被查詢人","相對人"]

ACTION_MAP = {
    "查詢": ["查詢","提供資料","提供保單資料","資料提供","查覆","查復","函覆","復文","函復"],
    "註記": ["註記","維持註記","延長註記","警示","限制"],
    "收取": ["收取","代收","逕收","就地收取","先行收取","先予收取"],
    "撤銷": ["撤銷","撤回","撤銷前令","撤銷命令"],
    "扣押": ["扣押","支付轉給命令","禁止給付","凍結","強制執行","執行命令"],
    "通知": ["通知","轉知","知會","請查照","如說明辦理"]
}

URGENT_TOKENS = ["急件","急速","速","儘速","即刻","從速","逕行","限期","逾期","刻不容緩","立即"]

# 欄位抽取
RE_SUBJECT = re.compile(r'(?:主旨|主文)\s*[:：]\s*(.+)')
RE_DESC1   = re.compile(r'(?:說明|理由)\s*[:：]\s*(?:一、|1[\.、])?\s*(.+?)(?:\n|。)', re.S)
RE_BASIS   = re.compile(r'(?:依據|依)\s*[:：]?\s*(.+?)(?:\n|。|；)')
RE_AGENCY  = re.compile(r'(?:發文機關|來文單位|機關|法院|執行處)\s*[:：]\s*([^\n\r，。；]+)')
RE_CASEID  = re.compile(r'((?:北|中|南|高|桃|新)?院[^\s，。:：]*?(?:司執|家執|智|字|年度)[^\s，。]*)')

# 保單號（簡化：6~20 位英數與 -）
RE_POLICY  = re.compile(r'(?:保單(?:號|編號)?)[：:\s]*([A-Za-z0-9\-]{6,20})')

# 身分證（TW），稍後會走校驗碼
RE_ID      = re.compile(r'\b([A-Z][12]\d{8})\b')

# 角色行：抓取「角色: 後面的整段」
ROLE_LINE = re.compile(r'(要保人|被保險人|受益人|債務人|債權人|公職人員)\s*[:：]\s*([^\n\r]+)')

# 遮罩姓名（王○○ / 張O○ / 李**）
NAME_MASK_RE = re.compile(r'^[\u4e00-\u9fff][○O＊\*]{1,3}$')

# 常見中文姓氏（不求完備，可再擴充）
SURNAME = set(list("陳林黃張李王吳劉蔡楊許鄭謝郭洪邱曾賴周葉蘇呂高潘朱簡鍾彭游江唐戴馮方宋施姚石杜侯阮薛盧梁趙顏余紀連馬董杜程溫藍蔣田魏涂鍾米金邵湯祁白夏錢邢尤"))

# 屏蔽詞（常出現在角色後、但不是姓名）
BANNED_TOKENS = {"名下","名下所有","清償","所有","本息","利息","保單","帳戶","扣押","限制",
                 "相關","等","之","部分","資料","請","應","逕","就地","先予","收取","撤銷","查詢","人員","先生","小姐"}

# 括號與切詞
SPLIT_RE   = re.compile(r'[，、；;。\s\(\)（）]+')
PARENS_RE  = re.compile(r'[\(（][^)\n\r（）]*[\)）]')

# 日期
RE_DATE_SLASH = re.compile(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})')         # 2025/09/05
RE_DATE_CHT   = re.compile(r'(?:中華民國)?\s*(\d{3,4})年(\d{1,2})月(\d{1,2})日')  # 114年9月5日 or 2025年9月5日
RE_WITHIN_DAYS = re.compile(r'(?:於|在)\s*(\d{1,3})\s*日內')

# ----------------------------
# 工具函式
# ----------------------------

def tw_roc_to_ad(y: int) -> int:
    return y + 1911 if y < 1911 else y

def chinese_numeral_to_int(s: str) -> Optional[int]:
    """
    極簡中文數字轉換（只處理 1~30 常見表述）
    """
    map1 = {"零":0,"〇":0,"一":1,"二":2,"兩":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
    if not s: return None
    if s in map1: return map1[s]
    # 十三、二十、二十五
    if s.endswith("十"):
        n = map1.get(s[0], 0)
        return 10 if n==0 else n*10
    if "十" in s:
        a,b = s.split("十")
        a = map1.get(a, 1)  # 十三 -> a=1
        b = map1.get(b, 0)
        return a*10 + b
    # 兩位純數字
    try:
        v = int(s)
        if 0 <= v <= 365: return v
    except:
        pass
    return None

def validate_twid(twid: str) -> bool:
    """
    台灣身分證號檢核碼
    格式: 1字母 + 1性別碼(1/2) + 8數字
    """
    if not re.fullmatch(r'[A-Z][12]\d{8}', twid): return False
    code = "ABCDEFGHJKLMNPQRSTUVXYWZIO"  # 沒有 'Z'？台灣使用此序列（含 I,O）
    n = code.index(twid[0]) + 10
    s1, s2 = divmod(n, 10)
    weights = [1,9,8,7,6,5,4,3,2,1]
    nums = [s1, s2] + [int(c) for c in twid[1:]]
    total = sum(w*n for w, n in zip(weights, nums))
    return total % 10 == 0

def redact_name(name: str) -> str:
    if not name: return name
    if len(name) == 2: return name[0] + "○"
    if len(name) == 3: return name[0] + "○" + name[2]
    return name[0] + "○"*(len(name)-2) + name[-1]

def redact_id(twid: str) -> str:
    return twid[:2] + "*" * 6 + twid[-2:]

def mask_list(vals: List[str], fn) -> List[str]:
    return [fn(v) for v in vals]

# ----------------------------
# 核心抽取邏輯
# ----------------------------

def _find_first(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    if not m: return ""
    try:
        return (m.group(1) or "").strip()
    except IndexError:
        return (m.group(0) or "").strip()

def extract_core_fields(text: str) -> Dict[str, str]:
    subject = _find_first(RE_SUBJECT, text)
    desc1   = _find_first(RE_DESC1, text)
    basis   = _find_first(RE_BASIS, text)
    agency  = _find_first(RE_AGENCY, text)
    caseid  = _find_first(RE_CASEID, text)
    return {"subject":subject, "desc1":desc1, "basis":basis, "agency":agency, "case_id":caseid}

def find_actions(text: str) -> List[Dict[str,str]]:
    acts = []
    for canon, variants in ACTION_MAP.items():
        for v in variants:
            if v in text:
                acts.append({"action": canon, "trigger": v})
                break
    return acts

def find_policies(text: str) -> List[str]:
    return list({m.group(1) for m in RE_POLICY.finditer(text)})

def find_ids(text: str) -> List[str]:
    ids = []
    for m in RE_ID.finditer(text):
        x = m.group(1)
        if validate_twid(x):
            ids.append(x)
    return list(dict.fromkeys(ids))  # 去重，保序

# ---- 姓名抽取（修正版） ----

def is_name_like(tok: str) -> bool:
    tok = tok.strip()
    if not tok:
        return False
    if NAME_MASK_RE.match(tok):  # 王○○ / 張** / 李O○
        return True
    if not (2 <= len(tok) <= 4):
        return False
    if any(c.isdigit() or ('A' <= c.upper() <= 'Z') for c in tok):
        return False
    if tok in BANNED_TOKENS:
        return False
    if tok[0] not in SURNAME:
        return False
    for c in tok[1:]:
        if not ('\u4e00' <= c <= '\u9fff' or c in '·‧・'):
            return False
    return True

def extract_name_from_role_segment(seg: str) -> str:
    """
    seg：如「王○○ 名下所有保單就地收取」或「李小明 清償」或「名下所有保單」。
    策略：去括號→切詞→優先遮罩名字→再用 name_like 判定。
    """
    clean = PARENS_RE.sub(" ", seg)
    toks = [t for t in (tok.strip() for tok in SPLIT_RE.split(clean)) if t]
    for t in toks:
        if NAME_MASK_RE.match(t):
            return t
    for t in toks:
        if is_name_like(t):
            return t
    return ""

def find_names(text: str) -> List[Dict[str,str]]:
    out = []
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        m = ROLE_LINE.search(ln)
        if not m:
            continue
        role, seg = m.group(1), m.group(2)
        cand = extract_name_from_role_segment(seg)

        # 若本行找不到，看看下一行（常換行）
        if not cand and i+1 < len(lines):
            cand = extract_name_from_role_segment(lines[i+1])

        # 若本行含身分證，盡量取 ID 左側最近的姓名樣式
        id_here = RE_ID.search(ln)
        if id_here:
            id_pos = id_here.start()
            tokens = [t for t in (tok.strip() for tok in SPLIT_RE.split(PARENS_RE.sub(" ", ln))) if t]
            left_tokens = [t for t in tokens if ln.find(t) < id_pos]
            for t in reversed(left_tokens):
                if NAME_MASK_RE.match(t) or is_name_like(t):
                    cand = t
                    break

        if cand:
            out.append({"role": role, "name": cand})
    # 去重（同角色多次，以首次為準）
    uniq = []
    seen = set()
    for x in out:
        k = (x["role"], x["name"])
        if k not in seen:
            uniq.append(x); seen.add(k)
    return uniq

# ---- 期限 / 急件 ----

def parse_due_date(text: str, today: Optional[dt.date]=None) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    回傳 (due_date_iso, reason_phrase, days_left)
    - 直接日期：YYYY/MM/DD 或 114年9月5日 / 2025年9月5日
    - 於X日內：支持阿拉伯數字與簡單中文數字（十、二十、二十五…）
    """
    today = today or dt.date.today()

    # 直接日期 2025/09/05
    m = RE_DATE_SLASH.search(text)
    if m:
        y, mm, dd = map(int, m.groups())
        try:
            d = dt.date(y, mm, dd)
            return d.isoformat(), m.group(0), (d - today).days
        except ValueError:
            pass

    # 中文日期 114年9月5日 / 2025年9月5日
    m = RE_DATE_CHT.search(text)
    if m:
        y = tw_roc_to_ad(int(m.group(1)))
        mm = int(m.group(2)); dd = int(m.group(3))
        try:
            d = dt.date(y, mm, dd)
            return d.isoformat(), m.group(0), (d - today).days
        except ValueError:
            pass

    # 於X日內（阿拉伯）
    m = RE_WITHIN_DAYS.search(text)
    if m:
        days = int(m.group(1))
        d = today + dt.timedelta(days=days)
        return d.isoformat(), m.group(0), days

    # 嘗試中文數字（例：於七日內 / 於二十五日內）
    m2 = re.search(r'(?:於|在)\s*([零〇一二兩三四五六七八九十]{1,3})\s*日內', text)
    if m2:
        days = chinese_numeral_to_int(m2.group(1))
        if days is not None:
            d = today + dt.timedelta(days=days)
            return d.isoformat(), m2.group(0), days

    return None, None, None

# ---- 摘要文字 ----

def build_summary_text(cls: str, agency: str, who: List[Dict[str,str]], actions: List[Dict[str,str]],
                       policies: List[str], due_phrase: Optional[str], due_date: Optional[str]) -> str:
    who_str = "、".join([f"{w.get('role','')} {w.get('name','')}".strip() for w in who]) if who else ""
    act_str = "、".join(sorted({a['action'] for a in actions})) if actions else ""
    pol_str = "、".join(policies[:3]) if policies else ""
    segs = []
    if agency: segs.append(f"{agency}")
    if cls:    segs.append(f"就「{cls}」來文")
    if who_str: segs.append(f"涉及 {who_str}")
    if pol_str: segs.append(f"關聯保單 {pol_str}")
    if act_str: segs.append(f"需辦事項：{act_str}")
    if due_phrase or due_date:
        if due_phrase and due_date:
            segs.append(f"{due_phrase}（至 {due_date}）")
        elif due_phrase:
            segs.append(due_phrase)
        else:
            segs.append(f"截至 {due_date}")
    return "；".join(segs) + "。"

# ----------------------------
# 主程式
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_csv", type=str, default="data/processed/test.csv",
                    help="含 doc_id、label 欄位；若沒有，請改指向 records.csv")
    ap.add_argument("--pred_csv", type=str, default="data/processed/test_predictions.csv",
                    help="含 doc_id、pred（預測類別），可無；若無則使用 label 當類別")
    ap.add_argument("--raw_root", type=str, default="raw", help="原始 txt 檔根目錄（底下是九個子資料夾）")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--fulltext_col", type=str, default="", help="若 records_csv 內含全文欄位，可指定如 FullText")
    ap.add_argument("--no_redact", action="store_true", help="不要脫敏姓名/身分證")
    ap.add_argument("--urgent_days", type=int, default=7, help="幾日內視為急件（預設 7）")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.records_csv)

    # 讀取 pred（若有）
    pred_map: Dict[str,str] = {}
    pred_path = Path(args.pred_csv)
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        if "doc_id" in pred_df.columns and "pred" in pred_df.columns:
            pred_map = {str(r.doc_id): str(r.pred) for _, r in pred_df.iterrows()}

    rows_out = []
    today = dt.date.today()

    for _, r in df.iterrows():
        doc_id = str(r.get("doc_id"))
        # 類別來源：pred_csv > label > ""
        label  = str(r.get("label")) if "label" in r else ""
        cls    = pred_map.get(doc_id, label)

        # 讀全文：優先 fulltext_col；否則 raw/<label>/<doc_id>.txt（找不到則全域搜尋一次）
        if args.fulltext_col and args.fulltext_col in df.columns:
            text = str(r.get(args.fulltext_col, "") or "")
        else:
            p = Path(args.raw_root) / (cls or label) / f"{doc_id}.txt"
            if not p.exists():
                cand = list(Path(args.raw_root).glob(f"**/{doc_id}.txt"))
                text = cand[0].read_text(encoding="utf-8", errors="ignore") if cand else ""
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")

        # ---- 抽取 ----
        core = extract_core_fields(text)
        ids  = find_ids(text)
        who  = find_names(text)
        acts = find_actions(text)
        pols = find_policies(text)
        due_date, due_phrase, days_left = parse_due_date(text, today=today)

        # ---- 急件判斷 ----
        urgent = False
        reasons = []
        if any(tok in text for tok in URGENT_TOKENS):
            urgent = True; reasons.append("命中急件詞")
        if days_left is not None and days_left <= args.urgent_days:
            urgent = True; reasons.append(f"{args.urgent_days}日內")

        # ---- 脫敏 ----
        if not args.no_redact:
            ids_out = mask_list(ids, redact_id)
            who_out = [{"role": w.get("role",""), "name": redact_name(w.get("name",""))} for w in who]
        else:
            ids_out = ids
            who_out = who

        # ---- 摘要 ----
        summary_text = build_summary_text(
            cls=cls or "",
            agency= core.get("agency",""),
            who=who_out,
            actions=acts,
            policies=pols,
            due_phrase=due_phrase,
            due_date=due_date
        )

        item = {
            "doc_id": doc_id,
            "class": cls,
            "urgency": {"is_urgent": bool(urgent), "reason": "；".join(reasons) if reasons else "", "due_date": due_date},
            "who": who_out,
            "ids": ids_out,
            "policies": pols,
            "basis": [core.get("basis","")] if core.get("basis") else [],
            "agency": core.get("agency",""),
            "case_id": core.get("case_id",""),
            "subject": core.get("subject",""),
            "summary_text": summary_text
        }
        rows_out.append(item)

    # ---- 輸出 JSONL ----
    jpath = out_dir / "summaries.jsonl"
    with jpath.open("w", encoding="utf-8") as f:
        for it in rows_out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # ---- 輸出 CSV（扁平檢視）----
    cpath = out_dir / "summaries.csv"
    flat = []
    for it in rows_out:
        flat.append({
            "doc_id": it["doc_id"],
            "class": it["class"],
            "is_urgent": it["urgency"]["is_urgent"],
            "due_date": it["urgency"]["due_date"],
            "agency": it["agency"],
            "case_id": it["case_id"],
            "ids": "；".join(it["ids"]),
            "who": "；".join([f"{w['role']}:{w['name']}" for w in it["who"]]),
            "policies": "；".join(it["policies"]),
            "basis": "；".join(it["basis"]),
            "subject": it["subject"],
            "summary_text": it["summary_text"]
        })
    pd.DataFrame(flat).to_csv(cpath, index=False, encoding="utf-8")

    print(f"[OK] 已輸出：\n- {jpath}\n- {cpath}")

if __name__ == "__main__":
    main()
