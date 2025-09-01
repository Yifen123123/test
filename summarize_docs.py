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

【任務要求】
1. 目的：萃取關鍵欄位，若沒有就填 null。
2. 僅輸出 JSON，不要文字解釋、不要加 ```json。
3. 欄位說明：
   - category: 原始類別（如：保單查詢/通知函/扣押命令…）
   - title: 10~40 字內中文摘要標題
   - summary: 100~200 字內要點摘要（誰／對誰／做什麼／何時／要求）
   - persons: 涉及之人名列表（中文名為主）
   - ids: 可能的身分證字號列表（格式：1 英文 + 9 數字）
   - policy_type: 保單種類（壽險/意外險/醫療險/傷害險/火險；若不明填 null）
   - policy_numbers: 保單或契約編號列表（英數 8–20）
   - actions: 本文涉及之動作/要求（如：查詢、撤銷、扣押、通知、補件）
   - date_mentions: 文中重要日期（yyyy-mm-dd 或民國紀年原樣）
   - parties: 涉及機關/公司/單位名列表
   - extra: 其他關鍵欄位（字典），如案件號、法院字號、金額、地址、電話等

【輸出 JSON 範例】
{{
  "category": "{category}",
  "title": "…",
  "summary": "…",
  "persons": ["…"],
  "ids": ["…"],
  "policy_type": "壽險",
  "policy_numbers": ["…"],
  "actions": ["查詢"],
  "date_mentions": ["民國114年01月01日"],
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

def validate_and_enhance(record: Dict[str, Any], category: str, subject: str, body: str) -> Dict[str, Any]:
    out = dict(record)

    # persons：若空，用啟發式補中文姓名候選
    persons = out.get("persons")
    if not isinstance(persons, list):
        persons = []
    candidates = CNAME_RE.findall(subject + "\n" + body)
    stop = {"附件", "說明", "主旨", "通知", "本院", "本局", "本公司", "承辦", "承辦人", "復文"}
    candidates = [c for c in candidates if c not in stop]
    out["persons"] = _unique_keep(list(persons) + candidates[:5])

    # ids：正則補
    ids = out.get("ids")
    if not isinstance(ids, list):
        ids = []
    out["ids"] = _unique_keep(list(ids) + TW_ID_RE.findall(subject + "\n" + body))

    # policy_numbers：正則補
    pols = out.get("policy_numbers")
    if not isinstance(pols, list):
        pols = []
    out["policy_numbers"] = _unique_keep(list(pols) + POLICY_RE.findall(subject + "\n" + body))

    # policy_type：缺時用關鍵詞補
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

    # actions：缺時用關鍵詞補
    if not out.get("actions"):
        s = subject + "\n" + body
        actions = []
        for k in ["查詢", "撤銷", "扣押", "通知", "補件", "更正", "函覆", "檢送", "轉知", "拍賣", "執行", "調查"]:
            if k in s:
                actions.append(k)
        out["actions"] = _unique_keep(actions) or None

    # extra：確保字典
    if not isinstance(out.get("extra"), dict):
        out["extra"] = {}

    # category 覆寫為來源（避免模型亂改）
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
