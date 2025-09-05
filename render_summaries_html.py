import argparse, json
from pathlib import Path
import pandas as pd

CLASS_COLORS = {
    "保單查詢": "#2563eb",
    "保單查詢＋註記": "#7c3aed",
    "保單註記": "#9333ea",
    "公職查詢": "#0891b2",
    "扣押命令": "#dc2626",
    "撤銷令": "#ea580c",
    "收取＋撤銷": "#b45309",
    "收取令": "#16a34a",
    "通知函": "#475569",
}

TEMPLATE = """<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公文摘要儀表板</title>
<style>
  :root {
    --bg:#0b1220; --panel:#121a2a; --muted:#94a3b8; --txt:#e2e8f0; --dim:#cbd5e1;
    --ok:#16a34a; --warn:#f59e0b; --bad:#ef4444; --chip:#1f2937;
  }
  body { margin:0; background:var(--bg); color:var(--txt); font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto; }
  header { padding:18px 22px; background:linear-gradient(180deg,#0f172a,#0b1220); border-bottom:1px solid #0f233d; }
  h1 { margin:0; font-size:18px; letter-spacing:1px; }
  .toolbar { display:flex; gap:12px; flex-wrap:wrap; margin-top:12px; }
  .toolbar input,.toolbar select { background:#0f172a; color:var(--txt); border:1px solid #1f2a44; border-radius:8px; padding:8px 10px; }
  .toolbar label { color:var(--muted); display:flex; align-items:center; gap:6px; }
  main { padding:20px; }
  table { width:100%; border-collapse:separate; border-spacing:0 8px; }
  thead th { text-align:left; font-weight:600; color:var(--dim); padding:8px 10px; cursor:pointer; }
  tbody tr { background:var(--panel); }
  tbody td { padding:10px; vertical-align:top; border-top:1px solid #12233b; border-bottom:1px solid #12233b; }
  tbody tr:hover { outline:1px solid #1e3a8a; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; color:#fff; }
  .badge { display:inline-block; padding:2px 8px; border-radius:6px; font-size:12px; }
  .urgent-true { background:rgba(239,68,68,.15); color:#fecaca; border:1px solid rgba(239,68,68,.35); }
  .urgent-false { background:rgba(148,163,184,.15); color:#e2e8f0; border:1px solid rgba(148,163,184,.3); }
  .duechip { padding:2px 6px; border-radius:6px; font-size:12px; }
  .due-soon { background:rgba(245,158,11,.18); color:#fde68a; }
  .due-now { background:rgba(239,68,68,.2); color:#fecaca; }
  .due-later { background:rgba(34,197,94,.18); color:#bbf7d0; }
  .chips span { background:var(--chip); color:#cbd5e1; padding:2px 6px; border-radius:6px; margin-right:6px; display:inline-block; }
  .muted { color:var(--muted); }
  .counts { margin:10px 0 0 2px; color:var(--muted); }
  a { color:#93c5fd; text-decoration: none; }
  a:hover { text-decoration: underline; }
</style>
</head>
<body>
<header>
  <h1>公文摘要儀表板</h1>
  <div class="toolbar">
    <input id="q" placeholder="關鍵字搜尋：主旨 / 摘要 / 機關 / 案號 / 人名 / 身分證…">
    <select id="cls">
      <option value="">全部類別</option>
    </select>
    <label><input type="checkbox" id="urgent"> 只看急件</label>
    <label>到期：<select id="dueWithin">
      <option value="">不限</option>
      <option value="3">3 天內</option>
      <option value="7" selected>7 天內</option>
      <option value="14">14 天內</option>
      <option value="30">30 天內</option>
    </select></label>
  </div>
  <div class="counts"><span id="count"></span></div>
</header>
<main>
  <table>
    <thead>
      <tr>
        <th data-k="doc_id">文件</th>
        <th data-k="class">類別</th>
        <th data-k="urgency">急件</th>
        <th data-k="due_date">期限</th>
        <th data-k="agency">機關</th>
        <th data-k="case_id">案號</th>
        <th>對象 / 身分證</th>
        <th>摘要</th>
      </tr>
    </thead>
    <tbody id="rows"></tbody>
  </table>
</main>

<script id="__DATA__" type="application/json">%%PAYLOAD%%</script>
<script>
const RAW_ROOT = %%RAW_ROOT%%;
const CLASS_COLORS = %%CLASS_COLORS%%;
const data = JSON.parse(document.getElementById("__DATA__").textContent || "[]");

// 轉換/歸一
const parseDate = s => s ? new Date(s + "T00:00:00") : null;
const today = new Date(); today.setHours(0,0,0,0);
data.forEach(d => {
  d._who_str = Array.isArray(d.who) ? d.who.map(w => (w.role||"") + (w.name?(":"+w.name):"")).join("；") : "";
  d._ids_str = Array.isArray(d.ids) ? d.ids.join("；") : (d.ids||"");
  d._pol_str = Array.isArray(d.policies) ? d.policies.join("；") : (d.policies||"");
  d._due = d.urgency && d.urgency.due_date ? parseDate(d.urgency.due_date) : null;
  d._days_left = d._due ? Math.ceil((d._due - today)/86400000) : null;
  d._urgent = d.urgency && d.urgency.is_urgent ? true : false;
});

// 填類別下拉
const clsSel = document.getElementById("cls");
[...new Set(data.map(d=>d.class).filter(Boolean))].forEach(c => {
  const opt = document.createElement("option"); opt.value=c; opt.textContent=c; clsSel.appendChild(opt);
});

// 排序
let sortKey = "doc_id", sortAsc = true;
document.querySelectorAll("thead th[data-k]").forEach(th => {
  th.addEventListener("click", () => {
    const k = th.getAttribute("data-k");
    if (sortKey===k) sortAsc = !sortAsc; else { sortKey=k; sortAsc=true; }
    render();
  });
});

// 篩選
document.getElementById("q").addEventListener("input", ()=>render());
document.getElementById("cls").addEventListener("change", ()=>render());
document.getElementById("urgent").addEventListener("change", ()=>render());
document.getElementById("dueWithin").addEventListener("change", ()=>render());

function passFilter(d) {
  const q = document.getElementById("q").value.trim();
  const cls = document.getElementById("cls").value;
  const urgentOnly = document.getElementById("urgent").checked;
  const dueWithin = document.getElementById("dueWithin").value;

  if (cls && d.class !== cls) return false;
  if (urgentOnly && !d._urgent) return false;

  if (dueWithin) {
    const days = parseInt(dueWithin,10);
    if (d._days_left == null || d._days_left > days) return false;
  }

  if (q) {
    const blob = (d.subject||"") + " " + (d.summary_text||"") + " " + (d.agency||"") + " " +
                 (d.case_id||"") + " " + d._who_str + " " + d._ids_str + " " + d._pol_str;
    if (!blob.toLowerCase().includes(q.toLowerCase())) return false;
  }
  return true;
}

function dueChip(d) {
  if (!d._due) return '<span class="muted">—</span>';
  const cls = d._days_left <= 0 ? "due-now" : (d._days_left <= 7 ? "due-soon" : "due-later");
  return `<span class="duechip ${cls}">${d.urgency?.due_date}（${d._days_left}天）</span>`;
}

function classPill(c) {
  const color = CLASS_COLORS[c] || "#334155";
  const style = `background:${color};`;
  return `<span class="pill" style="${style}">${c||"-"}</span>`;
}

function linkToRaw(d) {
  // raw/<class>/<doc_id>.txt （類別和檔名需編碼以避免 + 或空白）
  const cls = encodeURIComponent(d.class||"");
  const id = encodeURIComponent(d.doc_id||"");
  return RAW_ROOT + "/" + cls + "/" + id + ".txt";
}

function render() {
  let rows = data.filter(passFilter);
  // 排序
  rows.sort((a,b) => {
    let A=a[sortKey], B=b[sortKey];
    if (sortKey==="urgency") { A=a._urgent?1:0; B=b._urgent?1:0; }
    if (sortKey==="due_date") { A=a._due? a._due.getTime(): 9e15; B=b._due? b._due.getTime(): 9e15; }
    A = (A==null? "": A).toString(); B=(B==null?"":B).toString();
    return (A.localeCompare(B, 'zh-Hant', {numeric:true})) * (sortAsc?1:-1);
  });

  const tb = document.getElementById("rows");
  tb.innerHTML = rows.map(d => `
    <tr>
      <td>
        <div><strong>${d.doc_id||"-"}</strong></div>
        <div><a href="${linkToRaw(d)}" target="_blank">開啟原始檔</a></div>
      </td>
      <td>${classPill(d.class)}</td>
      <td><span class="badge ${d._urgent?'urgent-true':'urgent-false'}">${d._urgent?'急件':'—'}</span></td>
      <td>${dueChip(d)}</td>
      <td>${(d.agency||"-")
        .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}</td>
      <td>${(d.case_id||"-")
        .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}</td>
      <td>
        <div class="chips">${(d._who_str||"").split("；").filter(Boolean)
          .map(x=>`<span>${x.replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}</span>`).join(" ")}</div>
        <div class="muted">${(d._ids_str||"")
          .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}</div>
      </td>
      <td>
        <div><strong>${(d.subject||"")
          .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}</strong></div>
        <div class="muted" style="margin-top:4px">${(d.summary_text||"")
          .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")}</div>
      </td>
    </tr>
  `).join("");

  document.getElementById("count").textContent = `顯示 ${rows.length} / 總計 ${data.length} 筆`;
}

// 初始與預設排序
render();
</script>
</body>
</html>
"""

def load_data(path: Path):
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    else:
        raise SystemExit("只支援 .jsonl 或 .csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="summaries.jsonl 或 summaries.csv")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--raw_root", default="raw", help="原始 .txt 根目錄（用於產生檔案連結）")
    args = ap.parse_args()

    data = load_data(Path(args.input))
    payload = json.dumps(data, ensure_ascii=False)

    html = (TEMPLATE
            .replace("%%PAYLOAD%%", payload)
            .replace("%%RAW_ROOT%%", json.dumps(args.raw_root))
            .replace("%%CLASS_COLORS%%", json.dumps(CLASS_COLORS, ensure_ascii=False)))

    Path(args.out).write_text(html, encoding="utf-8")
    print(f"[OK] 已輸出：{args.out}")

if __name__ == "__main__":
    main()
