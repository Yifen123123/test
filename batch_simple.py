# app/batch.py
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from app.core.processor import process_file

app = FastAPI()

BASE = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE / "uploads"
OUTPUT_DIR = BASE / "outputs" / "json"
PROMPT = BASE / "system_prompt-v7.txt"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_json(text: str) -> Optional[Dict]:
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return None


def save_json(src_name: str, data: Dict) -> str:
    path = OUTPUT_DIR / f"{Path(src_name).stem}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.relative_to(BASE))


async def handle_one(file: UploadFile, sem: asyncio.Semaphore) -> Dict:
    async with sem:
        t0 = time.time()

        raw = await file.read()
        path = UPLOAD_DIR / file.filename
        path.write_bytes(raw)

        result: Union[str, Dict] = await asyncio.to_thread(
            process_file,
            file_path=str(path),
            prompt_path=str(PROMPT),
        )

        sec = round(time.time() - t0, 2)

        if isinstance(result, dict):
            data = result
        else:
            data = extract_json(result)

        if not data:
            return {
                "file": file.filename,
                "status": "fail",
                "seconds": sec,
            }

        return {
            "file": file.filename,
            "status": "ok",
            "seconds": sec,
            "json_file": save_json(file.filename, data),
        }


@app.post("/upload/batch")
async def batch(files: List[UploadFile] = File(...), workers: int = 1):
    sem = asyncio.Semaphore(max(1, workers))
    start = time.time()

    results = await asyncio.gather(*(handle_one(f, sem) for f in files))

    return JSONResponse({
        "total": len(results),
        "ok": sum(r["status"] == "ok" for r in results),
        "fail": sum(r["status"] == "fail" for r in results),
        "seconds": round(time.time() - start, 2),
        "items": results,
    })
