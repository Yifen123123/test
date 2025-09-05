# extract_and_build.py
import re
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from tqdm import tqdm

# -------- 正則（同義詞 + 欄位） --------
RE_SUBJECT = re.compile(r'(?:主旨|主文)\s*[:：]\s*(.+)')
RE_DESC    = re.compile(r'(?:說明|理由)\s*[:：]\s*(.+)', re.S)
RE_BASIS   = re.compile(r'(?:依據|依)\s*[:：]?\s*(.+?)(?:\n|。|；)')
RE_AGENCY  = re.compile(r'(?:發文機關|來文單位|機關|法院|執行處)\s*[:：]\s*([^\n\r，。；]+)')
RE_CASEID  = re.compile(r'((?:北|中|南|高|桃|新)?院[^\s，。:：]*?(?:司執|家執|智|字|年度)[^\s，。]*)')
RE_ADDRESS = re.compile(r'(?:地址)\s*[:：]\s*([^\n\r，。；]+)')
RE_OFFICER = re.compile(r'(?:承辦人)\s*[:：]\s*([^\n\r，。；]+)')
RE_PHONE   = re.compile(r'(?:電話|聯絡方式)\s*[:：]\s*([^\n\r，。；]+)')
RE_FAX     = re.compile(r'(?:傳真)\s*[:：]\s*([^\n\r，。；]+)')

RE_LAW     = re.compile(r'(民事訴訟法|強制執行法|保險法|行政執行法|個資法|金融消費者保護法)(第?\s*\d+\s*條)?')

def _find_first(pattern: re.Pattern, text: str) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    # 取第一個群組或整體
    try:
        return (m.group(1) or "").strip()
    except IndexError:
        return (m.group(0) or "").strip()

def extract_fields_from_text(text: str) -> Dict[str, str]:
    # 以全文嘗試抓各欄位
    subject = _find_first(RE_SUBJECT, text)
    desc    = _find_first(RE_DESC, text)
    basis   = _find_first(RE_BASIS, text)
    agency  = _find_first(RE_AGENCY, text)
    caseid  = _find_first(RE_CASEID, text)
    address = _find_first(RE_ADDRESS, text)
    officer = _find_first(RE_OFFICER, text)
    phone   = _find_first(RE_PHONE, text)
    fax     = _find_first(RE_FAX, text)

    law_m = RE_LAW.search(text)
    law = f"{law_m.group(1)}{law_m.group(2) or ''}".strip() if law_m else ""

    return {
        "subject": subject,
        "desc": desc,
        "basis": basis or law,   # basis 沒抓到就退而求其次用第一個法條
        "agency": agency,
        "case_id": caseid,
        "address": address,
        "officer": officer,
        "phone": phone,
        "fax": fax,
        "raw_text_len": len(text)
    }

def build_text(row: Dict[str, str], max_len: int = 300) -> str:
    """把抽出的欄位組合成短而密的輸入給 embedding/分類器"""
    parts = []
    if row.get("subject"): parts.append(f"[SUBJECT]{row['subject']}")
    if row.get("desc"):    parts.append(f"[DESC]{row['desc']}")
    if row.get("basis"):   parts.append(f"[BASIS]{row['basis']}")
    if row.get("agency"):  parts.append(f"[AGENCY]{row['agency']}")
    if row.get("case_id"): parts.append(f"[CASE]{row['case_id']}")
    # 基本資料通常對分類幫助小，但可留做檢索；若要加進模型，控制長度即可
    s = " / ".join(parts)
    return s[:max_len]

def from_csv(path: Path, text_col_map: Dict[str,str]) -> pd.DataFrame:
    """
    text_col_map: 指出各欄位所在欄名（若沒有就留空字串）
      例如：{"subject":"主旨", "desc":"說明", "basis":"依據", "agency":"發文機關",
             "case_id":"案號", "address":"地址", "officer":"承辦人", "phone":"電話", "fax":"傳真", "label":"label"}
    也支援只有全文欄位：{"fulltext":"全文"}
    """
    df = pd.read_csv(path)
    out_rows = []
    for i, r in df.iterrows():
        # 先走結構化欄位
        row = {
            "doc_id": r.get("doc_id", f"doc_{i}")
        }
        # 若有 fulltext，就直接用全文跑抽取；否則用現成欄位
        if text_col_map.get("fulltext"):
            fields = extract_fields_from_text(str(r.get(text_col_map["fulltext"], "")))
        else:
            fields = {
                "subject": str(r.get(text_col_map.get("subject",""), "") or ""),
                "desc":    str(r.get(text_col_map.get("desc",""), "") or ""),
                "basis":   str(r.get(text_col_map.get("basis",""), "") or ""),
                "agency":  str(r.get(text_col_map.get("agency",""), "") or ""),
                "case_id": str(r.get(text_col_map.get("case_id",""), "") or ""),
                "address": str(r.get(text_col_map.get("address",""), "") or ""),
                "officer": str(r.get(text_col_map.get("officer",""), "") or ""),
                "phone":   str(r.get(text_col_map.get("phone",""), "") or ""),
                "fax":     str(r.get(text_col_map.get("fax",""), "") or ""),
                "raw_text_len": 0
            }
            # 沒主旨/說明的情況，可嘗試從「主文/理由」欄位讀
            if not fields["subject"] and text_col_map.get("alt_subject"):
                fields["subject"] = str(r.get(text_col_map["alt_subject"], "") or "")
            if not fields["desc"] and text_col_map.get("alt_desc"):
                fields["desc"] = str(r.get(text_col_map["alt_desc"], "") or "")

        row.update(fields)
        row["label"] = r.get(text_col_map.get("label",""), None)
        row["built_text"] = build_text(row)
        out_rows.append(row)
    return pd.DataFrame(out_rows)

def from_txt_dir(txt_dir: Path) -> pd.DataFrame:
    out = []
    for i, p in enumerate(sorted(txt_dir.glob("*.txt"))):
        text = p.read_text(encoding="utf-8", errors="ignore")
        fields = extract_fields_from_text(text)
        row = {"doc_id": p.stem, **fields, "label": None}
        row["built_text"] = build_text(row)
        out.append(row)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--txt_dir", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="data/processed")
    # 下面對應你 csv 欄位名；沒有就留空字串
    ap.add_argument("--subject_col", type=str, default="主旨")
    ap.add_argument("--alt_subject_col", type=str, default="主文")
    ap.add_argument("--desc_col", type=str, default="說明")
    ap.add_argument("--alt_desc_col", type=str, default="理由")
    ap.add_argument("--basis_col", type=str, default="依據")
    ap.add_argument("--agency_col", type=str, default="發文機關")
    ap.add_argument("--case_id_col", type=str, default="案號")
    ap.add_argument("--address_col", type=str, default="地址")
    ap.add_argument("--officer_col", type=str, default="承辦人")
    ap.add_argument("--phone_col", type=str, default="電話")
    ap.add_argument("--fax_col", type=str, default="傳真")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--fulltext_col", type=str, default="")  # 若 CSV 僅有「全文」
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        text_col_map = {
            "subject": args.subject_col, "alt_subject": args.alt_subject_col,
            "desc": args.desc_col, "alt_desc": args.alt_desc_col,
            "basis": args.basis_col, "agency": args.agency_col, "case_id": args.case_id_col,
            "address": args.address_col, "officer": args.officer_col, "phone": args.phone_col, "fax": args.fax_col,
            "label": args.label_col, "fulltext": args.fulltext_col
        }
        df = from_csv(Path(args.csv), text_col_map)
    elif args.txt_dir:
        df = from_txt_dir(Path(args.txt_dir))
    else:
        raise SystemExit("請提供 --csv 或 --txt_dir")

    # 輸出 CSV + JSONL
    csv_path = out_dir / "records.csv"
    jsonl_path = out_dir / "records.jsonl"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(f"[OK] 已輸出 {len(df)} 筆")
    print(f"- CSV : {csv_path}")
    print(f"- JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()
