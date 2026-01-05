# app/batch.py
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.processor import process_file

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent          # project/app
PROJECT_DIR = BASE_DIR.parent                       # project/
UPLOAD_DIR = PROJECT_DIR / "uploads"
OUTPUT_JSON_DIR = PROJECT_DIR / "outputs" / "json"
PROMPT_PATH = PROJECT_DIR / "system_prompt-v7.txt"  # 你用 v7 就指到它

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

HR_TAG = "【人類可讀摘要】"
JSON_TAG = "【JSON_OUTPUT】"


# ---------------- helpers ----------------

def _split_human_and_json(text: str) -> Tuple[Optional[str], str]:
    """回傳 (human_readable, json_text)。若無雙區塊標籤，human_readable=None。"""
    if HR_TAG in text and JSON_TAG in text:
        human_part = text.split(HR_TAG, 1)[1].split(JSON_TAG, 1)[0].strip()
        json_part = text.split(JSON_TAG, 1)[1].strip()
        return human_part, json_part
    return None, text


def _extract_json_candidate(text: str) -> Optional[str]:
    """從文字中擷取最外層 {...} 片段。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start:end + 1]


def _safe_json_loads(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """回傳 (parsed_dict, error_message)。"""
    candidate = _extract_json_candidate(text)
    if not candidate:
        return None, "找不到 JSON 物件（未包含 '{' ... '}'）"
    try:
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            return None, "JSON 解析成功但不是物件（dict）"
        return parsed, None
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失敗：{e}"


def _save_json_for_filename(upload_filename: str, data: Dict[str, Any]) -> str:
    """存成 outputs/json/<同檔名>.json，回傳相對路徑字串。"""
    out_name = Path(upload_filename).with_suffix(".json").name
    out_path = OUTPUT_JSON_DIR / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(out_path.relative_to(PROJECT_DIR))


async def _process_one_file(
    file: UploadFile,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """
    單檔處理：
    - 存 uploads/
    - 呼叫 process_file
    - 拆雙區塊、parse JSON
    - 存 outputs/json
    """
    async with semaphore:
        t0 = time.time()

        # 1) 存檔
        saved_path = UPLOAD_DIR / file.filename
        content = await file.read()
        saved_path.write_bytes(content)

        # 2) 跑模型（thread，避免阻塞）
        raw_result: Union[str, Dict[str, Any]] = await asyncio.to_thread(
            process_file,
            file_path=str(saved_path),
            prompt_path=str(PROMPT_PATH),
        )

        seconds = round(time.time() - t0, 2)

        # 3) 若 processor 回 error dict
        if isinstance(raw_result, dict) and raw_result.get("error"):
            return {
                "filename": file.filename,
                "status": "fail",
                "error": raw_result["error"],
                "processing_time_seconds": seconds,
            }

        # 4) 解析（支援：雙區塊 or 純 JSON）
        human_readable: Optional[str] = None
        json_output: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        raw_text: Optional[str] = None

        if isinstance(raw_result, dict):
            json_output = raw_result
        else:
            raw_text = raw_result.strip()
            human_readable, json_text = _split_human_and_json(raw_text)
            json_output, parse_error = _safe_json_loads(json_text)

        # 5) 存檔
        json_file: Optional[str] = None
        if json_output:
            json_file = _save_json_for_filename(file.filename, json_output)

        resp: Dict[str, Any] = {
            "filename": file.filename,
            "status": "ok" if json_output else "fail",
            "processing_time_seconds": seconds,
            "human_readable": human_readable,
            "json_output": json_output,
            "json_file": json_file,
        }

        if parse_error:
            resp["error"] = parse_error
            # 只回一點 preview，避免回太大
            resp["raw_text_preview"] = (raw_text[:800] if raw_text else None)

        return resp


# ---------------- API ----------------

@app.post("/upload/batch")
async def upload_and_process_batch(
    files: List[UploadFile] = File(...),
    max_concurrency: int = 2,
):
    """
    批量上傳與處理：
    - files: 多檔
    - max_concurrency: 同時處理幾個（預設 2，3B 模型建議 1~2；太高反而更慢/更容易爆）
    """
    start_time = time.time()

    if max_concurrency < 1:
        return JSONResponse(status_code=400, content={"error": "max_concurrency 必須 >= 1"})

    semaphore = asyncio.Semaphore(max_concurrency)

    # 同時處理（受 semaphore 限制）
    tasks = [_process_one_file(f, semaphore) for f in files]
    results = await asyncio.gather(*tasks)

    ok = sum(1 for r in results if r.get("status") == "ok")
    fail = len(results) - ok

    total_seconds = round(time.time() - start_time, 2)

    return JSONResponse(
        content={
            "count": len(results),
            "ok": ok,
            "fail": fail,
            "max_concurrency": max_concurrency,
            "processing_time_seconds": total_seconds,
            "items": results,
        }
    )
