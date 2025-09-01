# -*- coding: utf-8 -*-
"""
make_briefs.py
將 summaries/*.jsonl（summarize_docs.py 的輸出）轉為：
  1) readable/briefs.html  卡片式清單（可搜尋/可折疊）
  2) readable/briefs.csv   匯總表（可給 Excel）

預設會遮蔽身分證與保單號；可用參數調整。

用法：
  python make_briefs.py --input-dir summaries --output-dir readable

可選參數：
  --no-mask-ids          不遮蔽身分證（預設遮蔽）
  --no-mask-policy       不遮蔽保單號（預設遮蔽）
  --max-summary 120      HTML 卡片上摘要顯示的最大字數（預設 120）
  --categories 保單查詢 通知函   只匯出特定類別
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

TW_ID_RE  = re.compile(r"^[A-Z][0-9]{9}$")
POLICY_RE = re.compile(r"^[A-Za-z0-9]{8,20}$")

def mask_tw_id(s: str) -> str:
    """A123456789 -> A123****89（保留前4後2）"""
    if not s or not TW_ID_RE.fullmatch(s):
        return s
    return s[:4] + "****" + s[-2:]

def mask_policy_no(s: str) -> str:
    """保單號遮蔽：前3後2，中間以*遮蔽，至少 4 個星號"""
    if not s or not POLICY_RE.fullmatch(s):
        return s
    if len(s) <= 5:
        return s[0] + "***" + s[-1]
    inner_len = max(len(s) - 5, 4)
    return s[:3] + ("*" * inner_len) + s[-2:]

def short(s: str, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + "…"

def html_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ 解析失敗 {path.name}: {e}")
    return out

def collect_records(input_dir: Path, categories: Optional[List[str]]) -> List[Dict[str, Any]]:
    files = sorted(input_dir.glob("*.jsonl"))
    if categories:
        allowed = set(categories)
        files = [f for f in files if f.stem in allowed]
    data: List[Dict[str, Any]] = []
    for f in files:
        data.extend(load_jsonl(f))
    # 排序：類別 → 檔名
    data.sort(key=lambda r: (str(r.get("category") or ""), str(r.get("filename") or "")))
    return data

def render_csv(rows: List[Dict[str, Any]], out_path: Path, mask_ids: bool, mask_policy: bool, max_summary: int):
    fields = [
        "filename", "category", "insurer", "title", "summary_short",
        "persons", "ids", "policy_numbers", "actions", "date_mentions", "doc_no"
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            persons = r.get("persons") or []
            # persons 可能是 [{"name","id_number"}] 或字串清單（保險）
            if persons and isinstance(persons[0], dict):
                names = [p.get("name") or "" for p in persons]
                ids   = [p.get("id_number") or "" for p in persons]
            else:
                names = persons if isinstance(persons, list) else []
                ids   = r.get("ids") or []

            if mask_ids:
                ids = [mask_tw_id(x) for x in ids]

            pols = r.get("policy_numbers") or []
            if mask_policy:
                pols = [mask_policy_no(x) for x in pols]

            writer.writerow({
                "filename": r.get("filename") or "",
                "category": r.get("category") or "",
                "insurer": r.get("insurer") or "",
                "title": r.get("title") or "",
                "summary_short": short(r.get("summary") or "", max_summary),
                "persons": "、".join(names),
                "ids": "、".join(ids),
                "policy_numbers": "、".join(pols),
                "actions": "、".join(r.get("actions") or []),
                "date_mentions": "、".join(r.get("date_mentions") or []),
                "doc_no": (r.get("extra") or {}).get("doc_no") or "",
            })
    print(f"✅ CSV 已輸出：{out_path}")

def render_html(rows: List[Dict[str, Any]], out_path: Path, mask_ids: bool, mask_policy: bool, max_summary: int):
    css = """
    body{font-family:system-ui,-apple-system,"Noto Sans TC",Segoe UI,Roboto,Helvetica,Arial; margin:24px; background:#f7f7f9}
    .toolbar{display:flex; gap:12px; align-items:center; margin-bottom:16px}
    .toolbar input{flex:1; padding:10px 12px; border:1px solid #ccc; border-radius:8px; font-size:14px}
    .badge{display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#eef; color:#225}
    .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px}
    .card{background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px; box-shadow:0 1px 2px rgba(0,0,0,.04)}
    .title{font-weight:600; margin:6px 0 8px; font-size:15px}
    .row{margin:4px 0; color:#333; font-size:14px}
    .label{color:#666; margin-right:6px}
    .summary{margin-top:8px; color:#222; background:#fafafa; border:1px dashed #e5e7eb; padding:8px; border-radius:8px; font-size:14px}
    .meta{display:flex; gap:8px; align-items:center; margin-bottom:6px}
    .pill{padding:2px 8px; border:1px solid #ddd; border-radius:999px; background:#fff; font-size:12px; color:#444}
    .muted{color:#888}
    .hidden{display:none}
    footer{margin-top:18px; color:#888; font-size:12px}
    """
    js = """
    const q = document.getElementById('q');
    const cards = [...document.querySelectorAll('.card')];
    q.addEventListener('input', () => {
      const kw = q.value.trim().toLowerCase();
      cards.forEach(c => {
        const hay = c.innerText.toLowerCase();
        c.style.display = hay.includes(kw) ? '' : 'none';
      });
    });
    """

    def format_persons(r: Dict[str, Any]) -> str:
        persons = r.get("persons") or []
        ids_flat = r.get("ids") or []
        items = []
        if persons and isinstance(persons[0], dict):
            for p in persons:
                name = p.get("name") or ""
                idn  = p.get("id_number")
                if idn and mask_ids: idn = mask_tw_id(idn)
                item = name if not idn else f"{name}（{idn}）"
                if item: items.append(item)
        else:
            # 退化處理
            names = persons if isinstance(persons, list) else []
            if mask_ids:
                ids_flat = [mask_tw_id(x) for x in ids_flat]
            for i, n in enumerate(names):
                pair = n
                if i < len(ids_flat) and ids_flat[i]:
                    pair += f"（{ids_flat[i]}）"
                items.append(pair)
        return "、".join([html_escape(x) for x in items])

    def format_policies(r: Dict[str, Any]) -> str:
        pols = r.get("policy_numbers") or []
        if mask_policy:
            pols = [mask_policy_no(x) for x in pols]
        return "、".join([html_escape(x) for x in pols])

    def card(r: Dict[str, Any]) -> str:
        category = html_escape(r.get("category") or "")
        insurer  = html_escape(r.get("insurer") or "")
        title    = html_escape(r.get("title") or "")
        summary  = html_escape(short(r.get("summary") or "", max_summary))
        persons  = format_persons(r)
        pols     = format_policies(r)
        actions  = "、".join([html_escape(x) for x in (r.get("actions") or [])]) or "—"
        dates    = "、".join([html_escape(x) for x in (r.get("date_mentions") or [])]) or "—"
        doc_no   = html_escape((r.get("extra") or {}).get("doc_no") or "—")
        fname    = html_escape(r.get("filename") or "—")
        court    = html_escape((r.get("extra") or {}).get("court") or "")

        meta = []
        if category: meta.append(f'<span class="pill">{category}</span>')
        if insurer:  meta.append(f'<span class="pill">{insurer}</span>')
        if court:    meta.append(f'<span class="pill">{court}</span>')
        meta_html = " ".join(meta)

        return f"""
        <div class="card">
          <div class="meta">{meta_html}</div>
          <div class="title">{title or "（無標題）"}</div>
          <div class="row"><span class="label">人員：</span>{persons or "—"}</div>
          <div class="row"><span class="label">保單：</span>{pols or "—"}</div>
          <div class="row"><span class="label">動作：</span>{actions}</div>
          <div class="row"><span class="label">日期：</span>{dates}</div>
          <div class="row"><span class="label">文號：</span>{doc_no}</div>
          <div class="row muted"><span class="label">來源：</span>{fname}</div>
          <div class="summary">{summary or "（無摘要）"}</div>
        </div>
        """

    cards_html = "\n".join(card(r) for r in rows)

    html = f"""<!doctype html>
<html lang="zh-Hant">
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>公文摘要清單</title>
<style>{css}</style>
<div class="toolbar">
  <input id="q" placeholder="輸入關鍵字搜尋：姓名、保單、公司、文號、摘要…" />
</div>
<div class="grid">
  {cards_html}
</div>
<footer>※ 本頁面為離線靜態檔案，支援關鍵字搜尋；敏感資訊已遮蔽（可在匯出時調整）。</footer>
<script>{js}</script>
</html>"""
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ HTML 已輸出：{out_path}")

def main():
    ap = argparse.ArgumentParser(description="把 summaries/*.jsonl 轉成給外勤看的 HTML＋CSV 總表")
    ap.add_argument("--input-dir", type=str, default="summaries", help="輸入資料夾（含 *.jsonl）")
    ap.add_argument("--output-dir", type=str, default="readable", help="輸出資料夾")
    ap.add_argument("--no-mask-ids", action="store_true", help="不要遮蔽身分證")
    ap.add_argument("--no-mask-policy", action="store_true", help="不要遮蔽保單號")
    ap.add_argument("--max-summary", type=int, default=120, help="HTML 卡片摘要最大字數")
    ap.add_argument("--categories", nargs="*", default=None, help="只匯出指定類別（檔名 stem）")
    args = ap.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_records(input_dir, args.categories)

    # 輸出 CSV 與 HTML
    csv_path  = output_dir / "briefs.csv"
    html_path = output_dir / "briefs.html"
    render_csv(rows, csv_path, mask_ids=not args.no_mask_ids, mask_policy=not args.no_mask_policy, max_summary=args.max_summary)
    render_html(rows, html_path, mask_ids=not args.no_mask_ids, mask_policy=not args.no_mask_policy, max_summary=args.max_summary)

if __name__ == "__main__":
    main()
