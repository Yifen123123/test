# evaluator.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


FIELDS = ("基準日", "來函機關", "收文編號", "查詢對象")


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def load_answer(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answer.json 頂層必須是 dict（key=檔名）")
    return {str(k).strip(): v for k, v in data.items()}


def extract_model_time(text: str) -> Tuple[str, str]:
    # 去 BOM + 全形空白 + 全形冒號轉半形冒號
    text = text.lstrip("\ufeff")
    model, time_str = None, None

    for raw in text.splitlines():
        line = raw.replace("\u3000", " ").strip().replace("：", ":")

        if model is None and line.startswith("模型:"):
            model = line.split("模型:", 1)[1].strip()

        if time_str is None and line.startswith("總執行時間:"):
            tail = line.split("總執行時間:", 1)[1].strip()
            m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", tail)
            if m:
                time_str = f"{int(m.group(1))}分{int(m.group(2)):02d}秒"
            else:
                time_str = tail  # 若格式不是 分秒，就原樣留著

        if model is not None and time_str is not None:
            break

    return model or "", time_str or "NA"


def parse_pred(path: Path) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    model, time_str = extract_model_time(text)

    # JSON 從第一個 '{' 開始
    i = text.find("{")
    if i == -1:
        raise ValueError(f"{path.name} 找不到 JSON 起點 '{{'")

    json_text = text[i:].strip().replace("“", '"').replace("”", '"')
    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} JSON 頂層必須是 dict（key=檔名）")

    data = {str(k).strip(): v for k, v in data.items()}
    # 模型真的抓不到才退回檔名（你要求不要檔名，這裡仍保底避免空白）
    model = model or path.stem
    return model, time_str, data


def evaluate(answer: Dict[str, Dict[str, Any]], pred: Dict[str, Dict[str, Any]]) -> Tuple[int, int]:
    keys = list(answer.keys())
    total_fields = len(keys) * len(FIELDS)
    mismatch = 0

    for k in keys:
        aobj = answer.get(k, {})
        pobj = pred.get(k, {})
        for f in FIELDS:
            if norm(aobj.get(f)) != norm(pobj.get(f)):
                mismatch += 1

    return mismatch, total_fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", required=True)
    ap.add_argument("--pred_dir", required=True)
    args = ap.parse_args()

    answer = load_answer(Path(args.answer))

    rows = []

    for p in sorted(Path(args.pred_dir).glob("*.json")):
        model, time_str, pred = parse_pred(p)
    
        mismatch, total_fields = evaluate(answer, pred)
        acc = 1 - mismatch / total_fields
    
        rows.append({
            "model": model,
            "acc": acc,
            "time_str": time_str
        })
    
    # 依正確率排序
    rows.sort(key=lambda r: r["acc"], reverse=True)
    
    # terminal 輸出
    for r in rows:
        print(
            f"{r['model']:<18} | acc={r['acc']:.3f} | time={r['time_str']}"
        )
    
    # 一次寫入 metrics.json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
