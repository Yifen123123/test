# -*- coding: utf-8 -*-
"""
summarize_docs.py
將 parsed/<category>.jsonl 的 subject/body/meta 餵給 LLM 做結構化摘要（抽取人名、身分證、保單種類、保單號碼等）。
支援 openai 風格與本地 ollama 兩種後端，並加入正則驗證與後處理。

用法：
  python summarize_docs.py --input-dir parsed --output-dir summaries --backend ollama --model qwen2.5:7b-instruct
  或
  OPENAI_API_KEY=xxx python summarize_docs.py --backend openai --model gpt-4o-mini
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# ----------------------------
# 1) 後端設定（選擇 openai 或 ollama）
# ----------------------------
BACKEND_CONFIG = {
    "backend": "ollama",            # "openai" 或 "ollama"
    "model": "qwen2.5:7b-instruct", # 例如：ollama 模型標籤；或 openai 的型號
    "temperature": 0.0,
}

# ----------------------------
# 2) 正則規則（台灣身分證、保單號等）
# ----------------------------
TW_ID_RE  = re.compile(r"\b[A-Z][0-9]{9}\b")             # 台灣身分證（簡單版）
POLICY_RE = re.compile(r"\b[A-Z0-9]{8,20}\b", re.IGNORECASE)  # 保單號（常見英數 8-20 位）
# 中文姓名（啟發式）：2~4 個漢字（含常見中點），僅作候選，最後用 LLM 結果交叉驗證
CNAME_RE  = re.compile(r"[\u4e00-\u9fa5·]{2,4}")

# ----------------------------
# 3) LLM 提示詞（指示輸出 JSON）
# ----------------------------
SYSTEM_PROMPT = """你是一個嚴格的資訊抽取器。你只輸出 JSON，絕不多說一句話。"""

# 針對「保單查詢」以及一般公文，給一個統一 schema（多類別可共用，缺的就 null）
USER_PROMPT_TEMPLATE = """請根據以下公文內容，產生結構化摘要（JSON 格式、UTF-8、strict JSON）。

【任務要求】
1. 目的：萃取關鍵欄位，若沒有就填 null。
2. 僅輸出 JSON，不要文字解釋。
3. 欄位說明：
   - category: 原始類別（如：保單查詢/通知函/扣押命令…）
   - title: 10~40 字內中文摘要標題（若無明確主旨，依內容擬定）
   - summary: 100~200 字內要點摘要（事件/主體/動作/時間/要求）
   - persons: 涉及之人名列表（中文名為主）
   - ids: 可能的身分證字號列表（格式檢核：1 英文 + 9 數字）
   - policy_type: 保單種類（如：壽險/意外險/醫療險/不明 → null）
   - policy_numbers: 保單或契約編號列表（英數 8–20）
   - actions: 本文涉及之動作/要求（例如：查詢、撤銷、扣押、通知、補件）
   - date_mentions: 文中重要日期（yyyy-mm-dd 或民國紀年原樣，不強制轉換）
   - parties: 涉及機關/公司/單位名列表
   - extra: 其他關鍵欄位（字典），如案件號、法院字號、金額、地址、電話等

【輸出 JSON 範例】
{{
  "category": "{category}",
  "title": "…",
  "summary": "…",
  "persons": ["…", "…"],
  "ids": ["…"],
  "policy_type": "壽險",
  "policy_numbers": ["…"],
  "actions": ["查詢"],
  "date_mentions": ["民國114年01月01日", "2025-08-15"],
  "parties": ["臺灣臺北地方法院", "全球人壽"],
  "extra": {{"doc_no": "北院執智字第123號"}}
}}

【文本】
<subject>
{subject}
</subject>

<body>
{body}
</body>
"""

# ----------------------------
# 4) 簡單的後處理與驗證
# ----------------------------
def _unique_keep(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s is None:
            continue
        s2 = str(s).strip()
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        out.append(s2)
    return out

def validate_and_enhance(record: Dict[str, Any], category: str, subject: str, body: str) -> Dict[str, Any]:
    """對 LLM 結果做正則校驗與增補（僅保守地修正/補齊，不做過度改寫）"""
    out = dict(record)

    # persons：若無，從文本用啟發式補候選（可能會有公務用詞誤擊，僅作輔助）
    persons = out.get("persons") or []
    if isinstance(persons, list) is False:
        persons = []
    # 從 subject/body 抓中文姓名候選
    candidates = CNAME_RE.findall(subject + "\n" + body)
    # 排除常見公文字眼（非常粗糙，可再擴充 stoplist）
    stop = {"附件", "說明", "主旨", "通知", "本院", "本局", "本公司", "承辦", "承辦人", "復文"}
    candidates = [c for c in candidates if c not in stop]
    persons = _unique_keep(list(persons) + candidates[:5])  # 不要太多，最多補 5 個
    out["persons"] = persons

    # ids：正則校驗台灣身分證
    ids = out.get("ids") or []
    if isinstance(ids, list) is False:
        ids = []
    regex_ids = TW_ID_RE.findall(subject + "\n" + body)
    # 交集優先：保留 LLM 有的 + 正則補的
    ids = _unique_keep(list(ids) + regex_ids)
    out["ids"] = ids

    # policy_numbers：正則補
    pols = out.get("policy_numbers") or []
    if isinstance(pols, list) is False:
        pols = []
    regex_pols = POLICY_RE.findall(subject + "\n" + body)
    # 排除明顯不像保單的短碼（例如 "DOCNO2025" 仍可能是文號，看情況保留）
    pols = _unique_keep(list(pols) + regex_pols)
    out["policy_numbers"] = pols

    # policy_type：若空，可以用關鍵詞啟發式補（非常簡單，可依你資料再擴充）
    if not out.get("policy_type"):
        s = subject + "\n" + body
        if "壽險" in s:
            out["policy_type"] = "壽險"
        elif "意外險" in s:
            out["policy_type"] = "意外險"
        elif "醫療險" in s or "醫療保險" in s:
            out["policy_type"] = "醫療險"
        elif "傷害險" in s:
            out["policy_type"] = "傷害險"
        elif "火險" in s:
            out["policy_type"] = "火險"
        else:
            out["policy_type"] = None

    # actions：若空，關鍵詞補
    if not out.get("actions"):
        actions = []
        s = subject + "\n" + body
        for k in ["查詢", "撤銷", "扣押", "通知", "補件", "更正", "函覆", "檢送", "轉知", "拍賣", "執行", "調查"]:
            if k in s:
                actions.append(k)
        out["actions"] = _unique_keep(actions) or None

    # extra：確保是 dict
    if not isinstance(out.get("extra"), dict):
        out["extra"] = {}

    # category 回寫（避免 LLM 改動）
    out["category"] = category

    return out

# ----------------------------
# 5) LLM 後端實作（openai 與 ollama）
# ----------------------------
def call_llm_openai(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    import requests
    api_key = BACKEND_CONFIG["openai_api_key"]
    api_base = BACKEND_CONFIG["openai_api_base"]
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未設定")

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

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
        "format": "json"  # 要求返回 JSON
    }
    r = requests.post(url, data=json.dumps(payload), timeout=300)
    r.raise_for_status()
    data = r.json()
    # Ollama 的 chat 回傳可能是流式；若非流式，data 應包含 "message"
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    # 若為非典型格式，直接回傳字串化
    return json.dumps(data, ensure_ascii=False)

def call_llm(category: str, subject: str, body: str) -> Dict[str, Any]:
    user_prompt = USER_PROMPT_TEMPLATE.format(category=category, subject=subject or "", body=body or "")
    backend = BACKEND_CONFIG["backend"]
    model = BACKEND_CONFIG["model"]
    temp = BACKEND_CONFIG["temperature"]

    if backend == "openai":
        raw = call_llm_openai(model, SYSTEM_PROMPT, user_prompt, temp)
    elif backend == "ollama":
        raw = call_llm_ollama(model, SYSTEM_PROMPT, user_prompt, temp)
    else:
        raise ValueError(f"未知 backend: {backend}")

    # 解析 JSON
    try:
        data = json.loads(raw)
    except Exception:
        # 若 LLM 偶爾吐出多餘字元，嘗試粗清理
        raw = raw.strip()
        # 嘗試截取最外層 JSON
        first = raw.find("{")
        last = raw.rfind("}")
        if first >= 0 and last >= 0 and last > first:
            raw2 = raw[first:last+1]
            data = json.loads(raw2)
        else:
            raise
    return data

# ----------------------------
# 6) 主流程：讀 parsed/*.jsonl -> 逐份文件摘要 -> 輸出 summaries/*.jsonl
# ----------------------------
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
        category = infile.stem  # 檔名 -> 類別
        outfile = output_dir / f"{category}.jsonl"
        n_ok, n_err = 0, 0

        with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if limit and i >= limit:
                    break
                try:
                    rec = json.loads(line)
                    subject = rec.get("subject", "") or ""
                    body = rec.get("body", "") or ""
                    llm_json = call_llm(category, subject, body)
                    final_json = validate_and_enhance(llm_json, category, subject, body)

                    # 附帶 filename 回寫，方便追蹤
                    final_json["filename"] = rec.get("filename")
                    # 也可把 meta.doc_no 拉到 extra 讓後續好用
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
    p = argparse.ArgumentParser(description="以 LLM 對公文做結構化摘要（含人名/身分證/保單資訊）")
    p.add_argument("--input-dir", type=str, default="parsed", help="輸入目錄（extract_body.py 的輸出）")
    p.add_argument("--output-dir", type=str, default="summaries", help="輸出目錄（每類別一個 .jsonl）")
    p.add_argument("--backend", type=str, choices=["openai", "ollama"], default=BACKEND_CONFIG["backend"])
    p.add_argument("--model", type=str, default=BACKEND_CONFIG["model"])
    p.add_argument("--categories", nargs="*", default=None, help="只處理指定類別（檔名 stem）")
    p.add_argument("--limit", type=int, default=None, help="每檔前 N 筆測試用")
    return p.parse_args()

def main():
    args = parse_args()
    BACKEND_CONFIG["backend"] = args.backend
    BACKEND_CONFIG["model"] = args.model

    print(f"🚀 backend={args.backend}  model={args.model}")
    process_all(Path(args.input_dir), Path(args.output_dir), args.categories, args.limit)
    print("🎉 完成")

if __name__ == "__main__":
    main()
