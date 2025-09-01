# -*- coding: utf-8 -*-
"""
summarize_docs.py (Ollama 專用 / 說明段落加強版)
------------------------------------------------
讀取 parsed/<category>.jsonl（每行一份：subject/body/meta/filename/category），
自動擷取「說明」段落，呼叫本地 Ollama 產生嚴格 JSON 的結構化摘要（公務員視角），
並做後處理（姓名/身分證/保單號/insurer/動作詞/期限補強、doc_no 回填）。

輸出 summaries/<category>.jsonl（每行一筆嚴格 JSON）。

用法：
  # 先確認 ollama 與模型（建議 3B 起步）：
  #   ollama serve
  #   ollama pull qwen2.5:3b-instruct
  #
  # 小量測試（1 筆、頭尾截斷、上下文 8192）：
  #   python summarize_docs.py --model qwen2.5:3b-instruct --limit 1 --truncate-mode headtail --num-ctx 8192
  #
  # 指定類別跑幾筆：
  #   python summarize_docs.py --model qwen2.5:3b-instruct --categories 保單查詢 --limit 5
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from urllib import request

# =========================
# 基本設定
# =========================

BACKEND_CONFIG = {
    "backend": "ollama",
    "model": "qwen2.5:3b-instruct",   # 可用 CLI 覆蓋
    "temperature": 0.0,
    "base_url": "http://127.0.0.1:11434",
}

MAX_CHARS_DEFAULT = 3000  # 正文截斷上限（可用 --max-chars 覆蓋）

# 正則：台灣身分證 / 保單號 / 中文姓名 / 期限
TW_ID_RE  = re.compile(r"\b[A-Z][0-9]{9}\b")
POLICY_RE = re.compile(r"\b[A-Z0-9]{8,20}\b", re.IGNORECASE)
CNAME_RE  = re.compile(r"[\u4e00-\u9fa5·]{2,4}")

# 期限（例：請於10日內、最遲於民國113年12月31日前）
DEADLINE_PATTERNS = [
    re.compile(r"(?:請於|應於|最遲於|限於)\s*([0-9]{1,2})\s*日內"),
    re.compile(r"(?:請於|應於|最遲於|限於)\s*(民國[0-9]{2,3}年[0-9]{1,2}月[0-9]{1,2}日)前?"),
    re.compile(r"(?:請於|應於|最遲於|限於)\s*([0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2})前?"),
    re.compile(r"(?:最遲|期限|限期)為?\s*(民國[0-9]{2,3}年[0-9]{1,2}月[0-9]{1,2}日|[0-9]{4}[/-][0-9]{1,2}[/-][0-9]{1,2})")
]

ROLE_STOPWORDS = {"債務人", "承辦", "承辦人", "被告", "申請人", "通知", "主旨", "說明", "附件", "本院", "本局", "本公司"}

KNOWN_INSURERS = [
    "全球人壽", "台灣人壽", "臺灣人壽", "國泰人壽", "新光人壽", "富邦人壽", "南山人壽", "中國人壽",
    "友邦人壽", "遠雄人壽", "宏泰人壽", "安聯人壽", "法國巴黎人壽", "保德信人壽"
]

# =========================
# Prompt：System / Schema / User
# =========================

SYSTEM_PROMPT = (
    "你是一位公務機關文書承辦視角的資訊抽取器。你只輸出 JSON（嚴格符合我提供的 schema），"
    "不可輸出其它任何文字、註解或Markdown圍欄。對於不確定的欄位，請輸出 null 或空陣列，不要猜測。"
)

SCHEMA_JSON = r'''{
  "category": "string",
  "title": "string",
  "summary": "string",
  "persons": [
    {"name": "string", "id_number": "string|null"}
  ],
  "policy_numbers": ["string"],
  "policy_type": "string|null",
  "insurer": "string|null",
  "actions": ["string"],
  "date_mentions": ["string"],

  "deadline": "string|null",
  "rationale_points": ["string"],
  "required_documents": ["string"],
  "agent_todo": ["string"],

  "extra": {
    "doc_no": "string|null",
    "court": "string|null",
    "insured_name": "string|null",
    "policy_holder": "string|null"
  }
}'''

USER_PROMPT_TEMPLATE = """請依下列【輸出規格】與【規則】，從公文文本萃取資訊，僅輸出一個 JSON 物件（公務員視角、用詞精確）。

【輸出規格(JSON schema)】
{SCHEMA_JSON}

【規則（務必遵守）】
1) 僅輸出 JSON，不得含有多餘文字或註解。
2) persons：只放「純姓名」，不可出現「債務人/承辦人/被告/申請人」等角色詞。
   - 若文本為「債務人王小明」，persons.name 只寫「王小明」。
   - 若找不到對應身分證，id_number=null；若找到，必須與該姓名配對。
   - 身分證必須符合 1 英文 + 9 數字（例如 A123456789），否則設為 null。
3) policy_numbers：只放保單/契約編號（英數 8–20 字元），排除法院文號/一般代碼。
4) insurer：填保險公司官方名稱（如：全球人壽/台灣人壽），若無則 null。
5) summary：60–160字，以「公文最主要用意」與「要求對象/動作」為核心（公務員視角）。
6) rationale_points：僅根據【說明節錄】的條列或文字，整理 1–5 點短句，保留原順序與關鍵事由/法源/事證。
7) agent_todo：以業務員實作視角，列出 1–5 條可執行待辦（動詞開頭），例如「查詢並提供XXX之保單資料」、「於期限前回覆法院」。
8) deadline：若文中出現「請於X日內」「最遲於YYYY/MM/DD前」等期限，原樣輸出；無則 null。
9) required_documents：若有「檢附/提供以下資料」等要求，整理為清單；無則空陣列。
10) actions：從文中抽取（查詢/撤銷/扣押/通知/補件/更正/函覆/檢送/轉知/執行/調查），沒有就空陣列。
11) 不確定時寧可設為 null 或空陣列，不要猜。

【範例（示意）】
<subject>關於查詢債務人保單資料</subject>
<body>本院通知：請全球人壽提供王小明（身分證A123456789）名下壽險保單PL20250101之相關資料，以利執行。</body>
<explain>
一、依強制執行需求，需確認債務人保單資產。
二、前案資料不足，請補齊。
</explain>

對應輸出：
{{
  "category": "{category}",
  "title": "法院請求提供王小明之壽險保單資料",
  "summary": "本院函請全球人壽提供王小明名下壽險保單PL20250101之資料，以配合法院強制執行程序。",
  "persons": [{{"name": "王小明", "id_number": "A123456789"}}],
  "policy_numbers": ["PL20250101"],
  "policy_type": "壽險",
  "insurer": "全球人壽",
  "actions": ["查詢", "通知"],
  "date_mentions": [],
  "deadline": null,
  "rationale_points": ["依強制執行程序需確認保單資產", "前案資料不足，需補齊"],
  "required_documents": ["王小明名下壽險保單PL20250101之基本資料與狀態"],
  "agent_todo": ["查詢王小明之PL20250101保單資料並彙整", "將資料回覆本院（依指定格式/管道）"],
  "extra": {{"doc_no": null, "court": "本院", "insured_name": "王小明", "policy_holder": null}}
}}

【待處理文本】
<subject>
{subject}
</subject>

<body>
{body}
</body>

【說明節錄（若無則空）】
<explain>
{explain}
</explain>
"""

# =========================
# 工具：HTTP、JSON 容錯、截斷、段落擷取
# =========================

def http_post_json(url: str, payload: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8"))

def safe_json_loads_loose(s: str) -> dict:
    s = s.strip()
    if s.startswith("```"):
        fence_end = s.find("```", 3)
        if fence_end != -1:
            inner = s[3:fence_end].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            s = inner.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    first = s.find("{")
    if first == -1:
        raise ValueError("No JSON object start '{' found.")
    stack = 0; in_str = False; esc = False; end_idx = -1
    for i, ch in enumerate(s[first:], start=first):
        if in_str:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': in_str = False
        else:
            if ch == '"': in_str = True
            elif ch == '{': stack += 1
            elif ch == '}':
                stack -= 1
                if stack == 0:
                    end_idx = i; break
    if end_idx == -1:
        raise ValueError("Unbalanced braces; cannot extract JSON object.")
    candidate = s[first:end_idx+1].strip()
    return json.loads(candidate)

def truncate_text(text: str, max_chars: int, mode: str = "headtail") -> str:
    if not text or len(text) <= max_chars:
        return text
    if mode == "head":
        return text[:max_chars] + f"\n…(已截斷，原長 {len(text)} 字)…"
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n…(中略，原長 {len(text)} 字，已截斷)…\n" + text[-tail:]

# --- 「說明」段落擷取：從 body 中切出「說明」到下一段標頭或結尾 ---
SECTION_HEAD_RE = re.compile(r"^\s*(主旨|說明|辦法|依據|法源|檢附|附件|注意事項|此致)\s*[:：]?\s*$", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*(?:[一二三四五六七八九十]\s*[、.]|[0-9]+\s*[).、]|•|-)\s*(.+)$", re.MULTILINE)

def extract_explain_segment(body: str) -> str:
    if not body:
        return ""
    # 找出所有段標頭位置
    matches = list(SECTION_HEAD_RE.finditer(body))
    if not matches:
        return ""
    # 找到最近一個「說明」標頭
    idx = None
    for i, m in enumerate(matches):
        if m.group(1) == "說明":
            idx = i
    if idx is None:
        return ""
    start = matches[idx].end()
    end = len(body)
    if idx + 1 < len(matches):
        end = matches[idx + 1].start()
    segment = body[start:end].strip()
    return segment

def extract_explain_bullets(segment: str, max_points: int = 6) -> List[str]:
    if not segment:
        return []
    bullets = [m.group(1).strip() for m in BULLET_RE.finditer(segment)]
    # 若沒偵測到條列，取前幾行做短句
    if not bullets:
        lines = [ln.strip(" 　") for ln in segment.splitlines() if ln.strip()]
        bullets = lines[:max_points]
    return bullets[:max_points]

# =========================
# Ollama 後端
# =========================

def call_llm_ollama(model: str, system: str, user: str, temperature: float = 0.0, num_ctx: Optional[int] = None) -> str:
    url = f"{BACKEND_CONFIG['base_url']}/api/chat"
    options: Dict[str, Any] = {"temperature": temperature}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "options": options,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "format": "json",
        "stream": False
    }
    for attempt in range(3):
        try:
            data = http_post_json(url, payload, timeout=600)
            msg = data.get("message") or {}
            content = msg.get("content", "")
            if not content and "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                content = data["messages"][-1].get("content", "")
            return content
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

# =========================
# 後處理：補強/校驗
# =========================

def _unique_keep(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq or []:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

def ensure_schema_keys(rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    # 既有
    out.setdefault("category", None)
    out.setdefault("title", ""); out.setdefault("summary", "")
    if not isinstance(out.get("persons"), list): out["persons"] = []
    if not isinstance(out.get("policy_numbers"), list): out["policy_numbers"] = []
    if "policy_type" not in out: out["policy_type"] = None
    if "insurer" not in out: out["insurer"] = None
    if not isinstance(out.get("actions"), list): out["actions"] = []
    if not isinstance(out.get("date_mentions"), list): out["date_mentions"] = []
    if not isinstance(out.get("extra"), dict): out["extra"] = {}
    out["extra"].setdefault("doc_no", None)
    out["extra"].setdefault("court", None)
    out["extra"].setdefault("insured_name", None)
    out["extra"].setdefault("policy_holder", None)
    # 新增
    out.setdefault("deadline", None)
    if not isinstance(out.get("rationale_points"), list): out["rationale_points"] = []
    if not isinstance(out.get("required_documents"), list): out["required_documents"] = []
    if not isinstance(out.get("agent_todo"), list): out["agent_todo"] = []
    return out

def validate_and_enhance(record: Dict[str, Any], category: str, subject: str, body: str) -> Dict[str, Any]:
    out = ensure_schema_keys(record)
    s = f"{subject}\n{body}"

    # persons：期望 [{"name","id_number"}]
    persons_in = out.get("persons", [])
    persons: List[Dict[str, Any]] = []
    for p in persons_in:
        if isinstance(p, dict):
            name = str(p.get("name", "")).strip()
            idn  = p.get("id_number", None)
            if name and name not in ROLE_STOPWORDS:
                if isinstance(idn, str):
                    idn = idn.strip()
                    if not TW_ID_RE.fullmatch(idn):
                        idn = None
                else:
                    idn = None if idn is not None else None
                persons.append({"name": name, "id_number": idn})
    # 若模型漏抓姓名，從文本補3個候選
    cand_names = [c for c in CNAME_RE.findall(s) if c not in ROLE_STOPWORDS]
    existing = {p["name"] for p in persons}
    for n in cand_names:
        if n not in existing and len(persons) < 3:
            persons.append({"name": n, "id_number": None})
            existing.add(n)
    out["persons"] = persons

    # policy_numbers：正則補
    pols = _unique_keep(list(out.get("policy_numbers", [])) + POLICY_RE.findall(s))
    out["policy_numbers"] = pols

    # policy_type 關鍵詞補
    if not out.get("policy_type"):
        if "壽險" in s: out["policy_type"] = "壽險"
        elif "意外險" in s: out["policy_type"] = "意外險"
        elif "醫療險" in s or "醫療保險" in s: out["policy_type"] = "醫療險"
        elif "傷害險" in s: out["policy_type"] = "傷害險"
        elif "火險" in s: out["policy_type"] = "火險"
        else: out["policy_type"] = None

    # actions 關鍵詞補
    if not out.get("actions"):
        acts = []
        for k in ["查詢","撤銷","扣押","通知","補件","更正","函覆","檢送","轉知","執行","調查"]:
            if k in s: acts.append(k)
        out["actions"] = _unique_keep(acts)

    # insurer：若缺則從正文補
    if not out.get("insurer"):
        for kw in KNOWN_INSURERS:
            if kw in s:
                out["insurer"] = kw; break

    # deadline：若缺則用正則補
    if not out.get("deadline"):
        dl = []
        for pat in DEADLINE_PATTERNS:
            for m in pat.finditer(s):
                g = [x for x in m.groups() if x]
                dl.extend(g)
        out["deadline"] = dl[0] if dl else None

    # category 覆寫來源
    out["category"] = category
    return out

# =========================
# 主流程
# =========================

def build_user_prompt(category: str, subject: str, body: str, explain: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        SCHEMA_JSON=SCHEMA_JSON,
        category=category,
        subject=subject or "",
        body=body or "",
        explain=explain or ""
    )

def call_llm(category: str, subject: str, body: str, explain: str, max_chars: int,
             truncate_mode: str = "headtail", num_ctx: Optional[int] = None) -> Dict[str, Any]:
    body_for_llm = truncate_text(body, max_chars, mode=truncate_mode) if body and len(body) > max_chars else (body or "")
    # 說明段保留較完整，但也做適度上限（避免爆 context）
    explain_trunc = truncate_text(explain, max(800, max_chars // 2), mode="head") if explain and len(explain) > max(800, max_chars // 2) else (explain or "")
    user_prompt = build_user_prompt(category, subject, body_for_llm, explain_trunc)

    raw = call_llm_ollama(
        model=BACKEND_CONFIG["model"],
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=BACKEND_CONFIG["temperature"],
        num_ctx=num_ctx,
    )
    try:
        data = safe_json_loads_loose(raw)
    except Exception as e:
        preview = raw[:400].replace("\n", "\\n")
        raise ValueError(f"LLM 回傳非純 JSON：{e}; 片段預覽={preview}")
    return data

def process_all(input_dir: Path, output_dir: Path,
                only_categories: Optional[List[str]] = None,
                limit: Optional[int] = None,
                max_chars: int = MAX_CHARS_DEFAULT,
                truncate_mode: str = "headtail",
                num_ctx: Optional[int] = None) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    files = sorted(input_dir.glob("*.jsonl"))

    if only_categories:
        allowed = set(only_categories)
        files = [f for f in files if f.stem in allowed]

    if not files:
        print(f"⚠️ 在 {input_dir} 找不到 .jsonl")
        return

    for infile in files:
        category = infile.stem
        outfile = output_dir / f"{category}.jsonl"
        n_ok, n_err = 0, 0

        with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if limit and i >= limit:
                    break
                try:
                    rec = json.loads(line)
                except Exception as e:
                    n_err += 1
                    print(f"❌ 輸入 JSON 解析失敗: {infile.name} line {i+1}: {e}")
                    continue

                try:
                    subject = rec.get("subject", "") or ""
                    body = rec.get("body", "") or ""
                    # 擷取「說明」段
                    explain_seg = extract_explain_segment(body)
                    # 如遇沒有標頭，試圖從條列推測說明
                    if not explain_seg:
                        # 若正文中條列很多，取前 6 條當作說明候選（保守）
                        bullets = extract_explain_bullets(body, max_points=6)
                        explain_seg = "\n".join(bullets)

                    llm_json = call_llm(
                        category, subject, body, explain_seg,
                        max_chars=max_chars, truncate_mode=truncate_mode, num_ctx=num_ctx
                    )
                    final_json = validate_and_enhance(llm_json, category, subject, body)

                    # 附加來源資訊
                    final_json["filename"] = rec.get("filename")
                    # 回填文號到 extra.doc_no（若有）
                    if "meta" in rec and isinstance(rec["meta"], dict):
                        doc_no = rec["meta"].get("doc_no")
                        if doc_no:
                            final_json.setdefault("extra", {})
                            final_json["extra"]["doc_no"] = doc_no

                    fout.write(json.dumps(final_json, ensure_ascii=False) + "\n")
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    print(f"❌ {infile.name} line {i+1}: {e}")

        print(f"✅ {category}: 成功 {n_ok} 筆，失敗 {n_err} 筆 -> {outfile}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="以本地 Ollama 對公文做結構化摘要（含說明段落、業務員待辦）")
    p.add_argument("--input-dir", type=str, default="parsed", help="輸入目錄（extract_body.py 的輸出）")
    p.add_argument("--output-dir", type=str, default="summaries", help="輸出目錄（每類別一個 .jsonl）")
    p.add_argument("--model", type=str, default=BACKEND_CONFIG["model"], help="Ollama 模型名稱，如 qwen2.5:3b-instruct")
    p.add_argument("--categories", nargs="*", default=None, help="只處理指定類別（檔名 stem），如：保單查詢 通知函")
    p.add_argument("--limit", type=int, default=None, help="每檔前 N 筆測試用")
    p.add_argument("--max-chars", type=int, default=MAX_CHARS_DEFAULT, help="正文最大字元數（超過會截斷）")
    p.add_argument("--truncate-mode", choices=["head", "headtail"], default="headtail",
                   help="截斷策略：head=只留開頭、headtail=保留頭尾（預設）")
    p.add_argument("--num-ctx", type=int, default=None,
                   help="Ollama context tokens（如 8192/16384，需模型支援）")
    return p.parse_args()

def main():
    args = parse_args()
    BACKEND_CONFIG["model"] = args.model
    print(f"🚀 backend=ollama  model={args.model}  max_chars={args.max_chars}  truncate_mode={args.truncate_mode}  num_ctx={args.num_ctx}")
    process_all(
        Path(args.input_dir),
        Path(args.output_dir),
        args.categories,
        args.limit,
        max_chars=args.max_chars,
        truncate_mode=args.truncate_mode,
        num_ctx=args.num_ctx,
    )
    print("🎉 完成")

if __name__ == "__main__":
    main()
