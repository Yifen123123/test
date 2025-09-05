# summarize_docs.py
import re
import json
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ---- 角色詞、動作詞、急件詞 ----
ROLE_TOKENS = ["要保人","被保險人","受益人","債務人","債權人","公職人員","被查詢人","相對人"]
URGENT_TOKENS = ["急件","急速","速","儘速","即刻","從速","逕行","限期","於","日內","逾期"]
ACTION_MAP = {
    "查詢": ["查詢","提供資料","提供保單資料","資料提供","查覆","查復","函覆","復文","函復"],
    "註記": ["註記","維持註記","延長註記","警示","限制"],
    "收取": ["收取","代收","逕收","就地收取","先行收取"],
    "撤銷": ["撤銷","撤回","撤銷前令","撤銷命令"],
    "扣押": ["扣押","支付轉給命令","禁止給付","凍結","強制執行","執行命令"],
    "通知": ["通知","轉知","知會","請查照","如說明辦理"]
}

# ---- 簡易正則 ----
RE_SUBJECT = re.compile(r'(?:主旨|主文)\s*[:：]\s*(.+)')
RE_DESC1   = re.compile(r'(?:說明|理由)\s*[:：]\s*(?:一、|1[\.、])?\s*(.+?)(?:\n|。)', re.S)
RE_BASIS   = re.compile(r'(?:依據|依)\s*[:：]?\s*(.+?)(?:\n|。|；)')
RE_AGENCY  = re.compile(r'(?:發文機關|來文單位|機關|法院|執行處)\s*[:：]\s*([^\n\r，。；]+)')
RE_CASEID  = re.compile(r'((?:北|中|南|高|桃|新)?院[^\s，。:：]*?(?:司執|家執|智|字|年度)[^\s，。]*)')
RE_POLICY  = re.compile(r'(?:保單(?:號|編號)?)[：:\s]*([A-Za-z0-9\-]{6,})')
RE_ID      = re.compile(r'\b([A-Z][12]\d{8})\b')   # 台灣身分證
RE_PHONE   = re.compile(r'(?:電話|聯絡方式)[：:\s]*([0-9\-#轉extEXT]{6,})')
RE_NAME_BY_ROLE = re.compile(r'(要保人|被保險人|受益人|債務人|債權人|公職人員)[：:\s]*([^\s，、。()（）]{2,4})')
RE_DATE_SLASH = re.compile(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})')         # 2025/09/05
RE_DATE_CHT   = re.compile(r'(\d{3,4})年(\d{1,2})月(\d{1,2})日')         # 114年9月5日 或 2025年9月5日
RE_WITHIN_DAYS = re.compile(r'於\s*(\d{1,3})\s*日內')

def tw_roc_to_ad(y: int) -> int:
    # 民國年轉西元
    return y + 1911 if y < 1911 else y

def parse_due_date(text: str, today: Optional[dt.date]=None) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """回傳 (due_date_iso, reason_phrase, days_left)"""
    today = today or dt.date.today()
    # 1) 直接日期
    m = RE_DATE_SLASH.search(text)
    if not m:
        m = RE_DATE_CHT.search(text)
        if m:
            y = tw_roc_to_ad(int(m.group(1))); mm = int(m.group(2)); dd = int(m.group(3))
            try:
                d = dt.date(y, mm, dd)
                return d.isoformat(), m.group(0), (d - today).days
            except ValueError:
                pass
    else:
        y, mm, dd = map(int, m.groups())
        try:
            d = dt.date(y, mm, dd)
            return d.isoformat(), m.group(0), (d - today).days
        except ValueError:
            pass
    # 2) 於X日內
    m = RE_WITHIN_DAYS.search(text)
    if m:
        days = int(m.group(1))
        d = today + dt.timedelta(days=days)
        return d.isoformat(), m.group(0), days
    return None, None, None

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
    return list({m.group(1) for m in RE_ID.finditer(text)})

def find_names(text: str) -> List[Dict[str,str]]:
    # 先從角色詞附近抓
    out = []
    for m in RE_NAME_BY_ROLE.finditer(text):
        out.append({"role": m.group(1), "name": m.group(2)})
    return out

def redact_name(name: str) -> str:
    if len(name) == 2:
        return name[0] + "○"
    if len(name) == 3:
        return name[0] + "○" + name[2]
    return name[0] + "○"*(len(name)-2) + name[-1]

def redact_id(twid: str) -> str:
    # A1******89
    return twid[:2] + "*"*6 + twid[-2:]

def mask_list(vals: List[str], fn) -> List[str]:
    return [fn(v) for v in vals]

def extract_core_fields(text: str) -> Dict:
    subject = (RE_SUBJECT.search(text) or [None, ""])[1].strip() if RE_SUBJECT.search(text) else ""
    desc1   = (RE_DESC1.search(text) or [None, ""])[1].strip() if RE_DESC1.search(text) else ""
    basis   = (RE_BASIS.search(text) or [None, ""])[1].strip() if RE_BASIS.search(text) else ""
    agency  = (RE_AGENCY.search(text) or [None, ""])[1].strip() if RE_AGENCY.search(text) else ""
    caseid  = (RE_CASEID.search(text) or [None, ""])[1].strip() if RE_CASEID.search(text) else ""
    return {"subject":subject, "desc1":desc1, "basis":basis, "agency":agency, "case_id":caseid}

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records_csv", type=str, default="data/processed/test.csv",
                    help="含 doc_id、label 欄位；若沒有，請改指向 records.csv")
    ap.add_argument("--pred_csv", type=str, default="data/processed/test_predictions.csv",
                    help="含 doc_id、pred（預測類別），可無；若無則使用 label 當類別")
    ap.add_argument("--raw_root", type=str, default="raw", help="原始 txt 檔根目錄（底下是九個子資料夾）")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--fulltext_col", type=str, default="", help="若 records_csv 內含全文欄位，可指定，如 FullText")
    ap.add_argument("--no_redact", action="store_true", help="不要脫敏姓名/身分證")
    ap.add_argument("--urgent_days", type=int, default=7, help="幾日內視為急件")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.records_csv)
    pred_map = {}
    pred_path = Path(args.pred_csv)
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
        pred_map = {r.doc_id: str(r.pred) for _, r in pred_df.iterrows() if "doc_id" in pred_df.columns and "pred" in pred_df.columns}

    rows_out = []
    today = dt.date.today()

    for _, r in df.iterrows():
        doc_id = str(r.get("doc_id"))
        label  = str(r.get("label")) if "label" in r else ""
        cls    = pred_map.get(doc_id, label)

        # 讀全文
        if args.fulltext_col and args.fulltext_col in df.columns:
            text = str(r.get(args.fulltext_col,"") or "")
        else:
            # raw/<label>/<doc_id>.txt
            p = Path(args.raw_root) / label / f"{doc_id}.txt"
            if not p.exists():
                # 允許在 records.csv 沒 label 的情況改從任何子資料夾尋找
                cand = list(Path(args.raw_root).glob(f"**/{doc_id}.txt"))
                text = cand[0].read_text(encoding="utf-8", errors="ignore") if cand else ""
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")

        core = extract_core_fields(text)
        ids  = find_ids(text)
        who  = find_names(text)
        acts = find_actions(text)
        pols = find_policies(text)
        due_date, due_phrase, days_left = parse_due_date(text, today=today)

        # 急件判斷
        urgent = False
        reason = ""
        if any(tok in text for tok in URGENT_TOKENS):
            urgent = True; reason = "命中急件詞"
        if days_left is not None and days_left <= args.urgent_days:
            urgent = True; reason = f"{args.urgent_days}日內"

        # 脫敏
        ids_out = ids if args.no_redact else mask_list(ids, redact_id)
        who_out = who if args.no_redact else [{"role": w.get("role",""), "name": redact_name(w.get("name",""))} for w in who]

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
            "urgency": {"is_urgent": bool(urgent), "reason": reason, "due_date": due_date},
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

    # 輸出
    jpath = out_dir / "summaries.jsonl"
    with jpath.open("w", encoding="utf-8") as f:
        for it in rows_out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # 也輸出 CSV（方便快速檢視）
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
