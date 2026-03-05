from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# 你要比對的四個欄位
FIELDS = ["基準日", "來函機關", "收文編號", "查詢對象"]

# meta 解析（注意你的 meta 在 """...""" 之內，但我們用 regex 直接抓行即可）
META_MODEL_RE = re.compile(r"^\s*模型：\s*(.+?)\s*$", re.M)
META_TIME_RE = re.compile(r"^\s*總執行時間：\s*([0-9]+)\s*分\s*([0-9]+)\s*秒\s*$", re.M)


def normalize_str(x: Any) -> str:
    """基本正規化：None -> '', 去頭尾空白、縮空白。"""
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_answer_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """答案檔必須是合法 JSON（純 JSON，沒有 meta 文字）。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("答案檔 JSON 頂層必須是 dict（key=檔名）")
    return data


def parse_prediction_file(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    你的預測檔格式：
    \"\"\" 
      資料：...
      模型：...
      總執行時間：3分10秒
    \"\"\"
    { ... JSON ... }

    我們會：
    - 用 regex 抽出 模型 / 時間
    - 找到第一個 '{' 後，把它當 JSON 本體 parse
    """
    text = path.read_text(encoding="utf-8")

    # 1) 抽 meta
    model = None
    time_str = None
    time_seconds = None

    m = META_MODEL_RE.search(text)
    if m:
        model = m.group(1).strip()

    t = META_TIME_RE.search(text)
    if t:
        minutes = int(t.group(1))
        seconds = int(t.group(2))
        time_seconds = minutes * 60 + seconds
        time_str = f"{minutes}分{seconds:02d}秒"

    # 2) 找 JSON 本體起點
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError(f"{path.name} 找不到 JSON 物件起點 '{{'")

    json_text = text[json_start:].strip()

    # 3) parse JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        # 提供更友善的錯誤訊息
        preview = json_text[:200].replace("\n", "\\n")
        raise ValueError(
            f"{path.name} JSON 解析失敗：{e}\n"
            f"JSON 片段預覽(前200字)：{preview}"
        )

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} JSON 頂層必須是 dict（key=檔名）")

    meta = {
        "model": model or path.stem,
        "time_str": time_str or "NA",
        "time_seconds": time_seconds if time_seconds is not None else -1,
    }
    return meta, data


@dataclass(frozen=True)
class ModelReport:
    file: str
    model: str
    time_str: str
    time_seconds: int
    total: int
    strict_correct: int
    strict_accuracy: float
    field_accuracy: float
    per_field_accuracy: Dict[str, float]
    missing_pred_keys: int
    extra_pred_keys: int


def evaluate_one_model(
    gt: Dict[str, Dict[str, Any]],
    pred: Dict[str, Dict[str, Any]],
) -> Tuple[int, int, float, float, Dict[str, float], int, int]:
    gt_keys = set(gt.keys())
    pred_keys = set(pred.keys())

    missing_pred = len(gt_keys - pred_keys)
    extra_pred = len(pred_keys - gt_keys)

    keys = sorted(gt_keys)
    total = len(keys)
    if total == 0:
        raise ValueError("答案檔沒有任何 case 可評測")

    strict_correct = 0
    per_field_hits = {f: 0 for f in FIELDS}
    total_field_hits = 0

    for k in keys:
        gt_obj = gt.get(k, {})
        pred_obj = pred.get(k, {})  # 若缺少這個 case，就當空 dict

        hits = 0
        for f in FIELDS:
            gt_v = normalize_str(gt_obj.get(f))
            pred_v = normalize_str(pred_obj.get(f))
            if gt_v == pred_v:
                hits += 1
                per_field_hits[f] += 1

        total_field_hits += hits
        if hits == len(FIELDS):
            strict_correct += 1

    strict_acc = strict_correct / total
    field_acc = total_field_hits / (total * len(FIELDS))
    per_field_acc = {f: per_field_hits[f] / total for f in FIELDS}

    return total, strict_correct, strict_acc, field_acc, per_field_acc, missing_pred, extra_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="答案檔（合法 JSON，純 JSON）")
    parser.add_argument("--pred_dir", required=True, help="預測檔資料夾（多個 .json，但檔內可含 meta + JSON）")
    parser.add_argument("--out", default="report.json", help="輸出報表檔名")
    args = parser.parse_args()

    answer_path = Path(args.answer)
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out)

    gt = load_answer_json(answer_path)

    reports: List[Dict[str, Any]] = []

    for p in sorted(pred_dir.glob("*.json")):
        meta, pred = parse_prediction_file(p)

        total, strict_correct, strict_acc, field_acc, per_field_acc, missing_pred, extra_pred = (
            evaluate_one_model(gt, pred)
        )

        r = ModelReport(
            file=p.name,
            model=meta["model"],
            time_str=meta["time_str"],
            time_seconds=meta["time_seconds"],
            total=total,
            strict_correct=strict_correct,
            strict_accuracy=round(strict_acc, 6),
            field_accuracy=round(field_acc, 6),
            per_field_accuracy={k: round(v, 6) for k, v in per_field_acc.items()},
            missing_pred_keys=missing_pred,
            extra_pred_keys=extra_pred,
        )
        reports.append(r.__dict__)

    # 排序：strict -> field
    reports.sort(key=lambda x: (x["strict_accuracy"], x["field_accuracy"]), reverse=True)

    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_path.resolve()}")
    for r in reports:
        print(
            f"{r['model']:<18} | strict={r['strict_accuracy']:.3f} "
            f"| field={r['field_accuracy']:.3f} | time={r['time_str']} "
            f"| missing={r['missing_pred_keys']} extra={r['extra_pred_keys']}"
        )


if __name__ == "__main__":
    main()
