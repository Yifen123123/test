# -*- coding: utf-8 -*-
"""
summarize_docs.py (Ollama 專用版)
---------------------------------
讀取 parsed/<category>.jsonl（每行一份文件：subject/body/meta/filename/category），
呼叫本地 Ollama 指令模型做「結構化摘要（只輸出 JSON）」，
並用正則做二次驗證/增補（人名、身分證、保單號）。

輸出 summaries/<category>.jsonl（每行一筆摘要 JSON）。

用法：
  # 先確認 ollama 服務與模型就緒
  # ollama serve
  # ollama pull qwen2.5:7b-instruct   （或先用較小模型 qwen2.5:1.5b-instruct）

  # 小量測試
  python summarize_docs.py --model qwen2.5:1.5b-instruct --limit 1

  # 指定類別
  python summarize_docs.py --model qwen2.5:7b-instruct --categories 保單查詢 --limit 5

  # 預設：
  #   input-dir = parsed
  #   output-dir = summaries
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

# =========================
# 基本設定
# =========================

BACKEND_CONFIG = {
    "backend": "ollama",               # 僅支援 ollama
    "model": "qwen2.5:7b-instruct",    # 依你本地拉的模型調整
    "temperature": 0.0,
}

# 長文截斷（避免本地小模型卡住）
MAX_CHARS = 3000

# 正則：台灣 ID / 保單號（簡化啟發式）
TW_ID_RE  = re.compile(r"\b[A-Z][0-9]{9}\b")
POLICY_RE = re.compile(r"\b[A-Z0-9]{8,20}\b", re.IGNORECASE)
CNAME_RE  = re.compile(r"[\u4e00-\u9fa5·]{2,4}")

# System 與 User Prompt（要求只輸出 JSON）
SYSTEM_PROMPT = "你是一個嚴格的資訊抽取器。你只輸出 JSON，絕不多說一句話。"

USER_PROMPT_TEMPLATE = """請根據以下公文內容，產生結構化摘要（JSON 格式、UTF-8、strict JSON）。

請依下列【輸出規格】與【規則】從公文文本萃取資訊，僅輸出一個 JSON 物件。

【輸出規格(JSON schema)】
{SCHEMA_JSON}

【規則（務必遵守）】
1) 僅輸出 JSON，不得含有多餘文字或註解。
2) persons：只放「純姓名」，不可出現「債務人/承辦人/被告/申請人」等角色詞。
   - 若文本為「債務人王小明」，persons.name 只寫「王小明」。
   - 若找不到對應身分證，id_number=null；若找到，必須與該姓名配對。
   - 字母+9數字（如 A123456789）才算身分證；錯一碼就不要填。
3) policy_numbers：只放保單/契約編號（英數8–20字元），排除法院文號/一般代碼。
4) insurer：填保險公司官方名稱（如：全球人壽、台灣人壽），若無則 null。
5) summary：60–160字，說明此公文「最主要用意」與「要求對象/動作」。
6) actions：從文中抽取（查詢/撤銷/扣押/通知/補件/更正/函覆/檢送/轉知/執行/調查），沒有就空陣列。
7) 不確定時寧可設為 null 或空陣列，不要猜。

【範例1（示意）】
<subject>關於查詢債務人保單資料</subject>
<body>本院通知：請全球人壽提供王小明（身分證A123456789）名下壽險保單PL20250101之相關資料，以利執行。</body>

對應輸出：
{{
  "category": "{category}",
  "title": "法院請求提供王小明之壽險保單資料",
  "summary": "本院通知全球人壽，請提供王小明名下壽險保單PL20250101之資料，以配合法院執行作業。",
  "persons": [{{"name": "王小明", "id_number": "A123456789"}}],
  "policy_numbers": ["PL20250101"],
  "policy_type": "壽險",
  "insurer": "全球人壽",
  "actions": ["查詢", "通知"],
  "date_mentions": [],
  "extra": {{"doc_no": null, "court": "本院", "insured_name": "王小明", "policy_holder": null}}
}}

【範例2（示意）】
<subject>函請提供契約編號 QX998877</subject>
<body>茲依申請程序，請 貴公司提供要保人李小美名下之契約QX998877資料。未載明身分證。</body>

對應輸出：
{{
  "category": "{category}",
  "title": "請提供李小美名下契約資料",
  "summary": "函請保險公司提供要保人李小美女士名下契約QX998877之資料，俾便案件審查。",
  "persons": [{{"name": "李小美", "id_number": null}}],
  "policy_numbers": ["QX998877"],
  "policy_type": null,
  "insurer": null,
  "actions": ["查詢"],
  "date_mentions": [],
  "extra": {{"doc_no": null, "court": null, "insured_name": null, "policy_holder": "李小美"}}
}}

【待處理文本】
<subject>
{subject}
</subject>

<body>
{body}
</body>
"""

# =========================
# 工具：鬆容錯 JSON 解析
# =========================

def safe_json_loads_loose(s: str) -> dict:
    """
    盡力從 s 中擷取第一個合法 JSON 物件：
      - 移除 ```json ... ``` 或 ``` ... ```
      - 找出第一段「大括號平衡」的片段
      - 忽略 JSON 外的雜訊
    失敗則丟 ValueError
    """
    s = s.strip()

    # 去除圍欄 ```json / ``` 包裹
    if s.startswith("```"):
        fence_end = s.find("```", 3)
        if fence_end != -1:
            inner = s[3:fence_end].strip()
            if inner.lower().startswith("json"):
                inner = inner[4:].strip()
            s = inner.strip()

    # 直接嘗試
    try:
        return json.loads(s)
    except Exception:
        pass

    # 從第一個 { 起，做括號平衡
    first = s.find("{")
    if first == -1:
        raise ValueError("No JSON object start '{' found.")

    stack = 0
    in_str = False
    esc = False
    end_idx = -1
    for i, ch in enumerate(s[first:], start=first):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                stack += 1
            elif ch == '}':
                stack -= 1
                if stack == 0:
                    end_idx = i
                    break

    if end_idx == -1:
        raise ValueError("Unbalanced braces; cannot extract JSON object.")

    candidate = s[first:end_idx+1].strip()
    return json.loads(candidate)

# =========================
# Ollama 後端
# =========================

def call_llm_ollama(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    import requests
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "format": "json",
        "stream": False  # 非串流，回完整 JSON
    }

    # 簡單重試（下載/載入模型時較慢）
    for attempt in range(3):
        try:
            resp = requests.post(url, data=json.dumps(payload), timeout=600)
            resp.raise_for_status()
            data = resp.json()
            msg = data.get("message") or {}
            content = msg.get("content", "")
            if not content and "messages" in data and isinstance(data["messages"], list) and data["messages"]:
                content = data["messages"][-1].get("content", "")
            return content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))

# =========================
# 後處理：正則校驗/增補
# =========================

def _unique_keep(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq or []:
        s = str(s).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

ROLE_STOPWORDS = {"債務人", "承辦", "承辦人", "被告", "申請人", "通知", "主旨", "說明", "附件", "本院", "本局", "本公司"}

def validate_and_enhance(record, category, subject, body):
    out = dict(record)
    s = f"{subject}\n{body}"

    # persons：期望為 [{"name":..., "id_number":...}]
    persons = out.get("persons")
    if not isinstance(persons, list):
        persons = []
    # 先把 LLM 給的資料清洗：去掉空白與非法型別
    clean = []
    for p in persons:
        if isinstance(p, dict):
            name = str(p.get("name", "")).strip()
            idn  = p.get("id_number", None)
            if name and name not in ROLE_STOPWORDS:
                # 身分證正則校驗
                if isinstance(idn, str) and not TW_ID_RE.fullmatch(idn.strip()):
                    idn = None
                clean.append({"name": name, "id_number": idn})
    persons = clean

    # 用啟發式補姓名候選（無配對就 id_number=null）
    cand_names = [c for c in CNAME_RE.findall(s) if c not in ROLE_STOPWORDS]
    # 去除已存在的姓名
    existing = {p["name"] for p in persons}
    for n in cand_names:
        if n not in existing and len(persons) < 5:  # 控制最多補 5 個
            persons.append({"name": n, "id_number": None})
            existing.add(n)
    out["persons"] = persons

    # ids 與 policy_numbers：維持既有欄位，但我們優先信任 LLM；正則只做補充
    ids = out.get("ids")
    if not isinstance(ids, list):
        ids = []
    ids = _unique_keep(list(ids) + TW_ID_RE.findall(s))
    out["ids"] = ids

    pols = out.get("policy_numbers")
    if not isinstance(pols, list):
        pols = []
    pols = _unique_keep(list(pols) + POLICY_RE.findall(s))
    out["policy_numbers"] = pols

    # policy_type：關鍵詞補
    if not out.get("policy_type"):
        if "壽險" in s: out["policy_type"] = "壽險"
        elif "意外險" in s: out["policy_type"] = "意外險"
        elif "醫療險" in s or "醫療保險" in s: out["policy_type"] = "醫療險"
        elif "傷害險" in s: out["policy_type"] = "傷害險"
        elif "火險" in s: out["policy_type"] = "火險"
        else: out["policy_type"] = None

    # actions：關鍵詞補
    if not out.get("actions"):
        actions = []
        for k in ["查詢","撤銷","扣押","通知","補件","更正","函覆","檢送","轉知","執行","調查"]:
            if k in s: actions.append(k)
        out["actions"] = _unique_keep(actions) or []

    # insurer：若模型沒給，試簡單抽公司名
    if not out.get("insurer"):
        for kw in ["全球人壽","台灣人壽","國泰人壽","新光人壽","富邦人壽","南山人壽"]:
            if kw in s:
                out["insurer"] = kw
                break
        else:
            out["insurer"] = None

    # extra：確保字典，回填 doc_no
    if not isinstance(out.get("extra"), dict):
        out["extra"] = {}
    # 回填來源 doc_no（extract_body 產出的 meta.doc_no）
    # 此步驟保留在主流程中已有，你可保留或移到這裡

    # category 覆寫
    out["category"] = category
    return out

# =========================
# 主流程
# =========================

def call_llm(category: str, subject: str, body: str) -> Dict[str, Any]:
    if body and len(body) > MAX_CHARS:
        body = body[:MAX_CHARS] + "\n…(截斷)…"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        category=category,
        subject=subject or "",
        body=body or ""
    )
    raw = call_llm_ollama(
        model=BACKEND_CONFIG["model"],
        system=SYSTEM_PROMPT,
        user=user_prompt,
        temperature=BACKEND_CONFIG["temperature"],
    )
    try:
        data = safe_json_loads_loose(raw)
    except Exception as e:
        preview = raw[:400].replace("\n", "\\n")
        raise ValueError(f"LLM 回傳非純 JSON：{e}; 片段預覽={preview}")
    return data

def process_all(input_dir: Path, output_dir: Path, only_categories: Optional[List[str]] = None, limit: Optional[int] = None):
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
                    llm_json = call_llm(category, subject, body)
                    final_json = validate_and_enhance(llm_json, category, subject, body)

                    # 附加來源資訊
                    final_json["filename"] = rec.get("filename")
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
    p = argparse.ArgumentParser(description="以本地 Ollama 對公文做結構化摘要（人名/身分證/保單資訊）")
    p.add_argument("--input-dir", type=str, default="parsed", help="輸入目錄（extract_body.py 的輸出）")
    p.add_argument("--output-dir", type=str, default="summaries", help="輸出目錄（每類別一個 .jsonl）")
    p.add_argument("--model", type=str, default=BACKEND_CONFIG["model"], help="Ollama 模型名稱，如 qwen2.5:7b-instruct")
    p.add_argument("--categories", nargs="*", default=None, help="只處理指定類別（檔名 stem），如：保單查詢 通知函")
    p.add_argument("--limit", type=int, default=None, help="每檔前 N 筆測試用")
    return p.parse_args()

def main():
    args = parse_args()
    BACKEND_CONFIG["model"] = args.model

    print(f"🚀 backend=ollama  model={args.model}")
    process_all(Path(args.input_dir), Path(args.output_dir), args.categories, args.limit)
    print("🎉 完成")

if __name__ == "__main__":
    main()
