# evaluator.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple


FIELDS = ("基準日", "來函機關", "收文編號", "查詢對象")


def norm(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).strip())


def load_answer(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("answer.json 頂層必須是 dict（key=檔名）")
    return {k.strip(): v for k, v in data.items()}


def parse_pred(path: Path) -> Tuple[str, str, Dict[str, Dict[str, Any]]]:
    """
    讀取 batch_output_XX.json：
    - 抽 meta：模型 / 總執行時間
    - 抽 JSON：從第一個 { 開始 json.loads
    """
    text = path.read_text(encoding="utf-8")

    # 抽模型
    m = re.search(r"模型：\s*(.+)", text)
    model = m.group(1).strip() if m else path.stem

    # 抽時間（3分10秒）
    t = re.search(r"總執行時間：\s*(\d+)\s*分\s*(\d+)\s*秒", text)
    time_str = f"{int(t.group(1))}分{int(t.group(2)):02d}秒" if t else "NA"

    # 抽 JSON 本體
    i = text.find("{")
    if i == -1:
        raise ValueError(f"{path.name} 找不到 JSON 起點 '{{'")

    json_text = text[i:].strip()

    # 若你的 LLM 偶爾輸出智慧引號，這行能救命（不加長度也很值得）
    json_text = json_text.replace("“", '"').replace("”", '"')

    data = json.loads(json_text)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} JSON 頂層必須是 dict（key=檔名）")

    data = {k.strip(): v for k, v in data.items()}
    return model, time_str, data


def evaluate(answer: Dict[str, Dict[str, Any]], pred: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    akeys = set(answer.keys())
    pkeys = set(pred.keys())

    missing = len(akeys - pkeys)  # 答案有，預測沒有
    extra = len(pkeys - akeys)    # 預測有，答案沒有

    mismatch_fields = 0
    total_fields = len(akeys) * len(FIELDS)

    for k in akeys:
        aobj = answer.get(k, {})
        pobj = pred.get(k, {})  # 缺失就當全部錯
        for f in FIELDS:
            if norm(aobj.get(f)) != norm(pobj.get(f)):
                mismatch_fields += 1

    field_acc = 0.0 if total_fields == 0 else (1 - mismatch_fields / total_fields)
    return {
        "cases": len(akeys),
        "total_fields": total_fields,
        "mismatch_fields": mismatch_fields,
        "field_accuracy": field_acc,
        "missing": missing,
        "extra": extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer", required=True)
    ap.add_argument("--pred_dir", required=True)
    args = ap.parse_args()

    answer = load_answer(Path(args.answer))
    pred_dir = Path(args.pred_dir)

    rows = []
    for p in sorted(pred_dir.glob("*.json")):
        model, time_str, pred = parse_pred(p)
        score = evaluate(answer, pred)
        rows.append((model, time_str, score))

    # 依 mismatch 少的（越好）排序
    rows.sort(key=lambda x: x[2]["mismatch_fields"])

    for model, time_str, s in rows:
        print(
            f"{model:<18} | mismatch={s['mismatch_fields']}/{s['total_fields']} "
            f"| acc={s['field_accuracy']:.3f} | time={time_str} "
            f"| missing={s['missing']} extra={s['extra']}"
        )


if __name__ == "__main__":
    main()
