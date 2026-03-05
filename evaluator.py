from __future__ import annotations

import json
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List


# =========================
# Meta parsing (first 4 lines)
# =========================
META_MODEL_RE = re.compile(r"^\s*模型：\s*(.+?)\s*$")
META_TIME_RE = re.compile(r"^\s*總執行時間：\s*([0-9]+)\s*分\s*([0-9]+)\s*秒\s*$")


# 你要比對的欄位（以答案為準也可以，但固定欄位更安全）
FIELDS = ["基準日", "來函機關", "收文編號", "查詢對象"]


def normalize_str(x: Any) -> str:
    """做最基本的正規化：None -> '', 去頭尾空白、縮空白。"""
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_meta_and_json(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    模型輸出檔：
      前四行是 meta（文字）
      第五行開始是 JSON（dict keyed-by-filename）
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    meta_lines = lines[:4]
    json_text = "\n".join(lines[4:]).strip()

    model_name: Optional[str] = None
    time_str: Optional[str] = None
    time_seconds: Optional[int] = None

    for line in meta_lines:
        m = META_MODEL_RE.match(line)
        if m:
            model_name = m.group(1).strip()

        t = META_TIME_RE.match(line)
        if t:
            minutes = int(t.group(1))
            seconds = int(t.group(2))
            time_seconds = minutes * 60 + seconds
            time_str = f"{minutes}分{seconds:02d}秒"

    if not json_text:
        raise ValueError(f"{path.name} 第五行後找不到 JSON 內容")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name} JSON 解析失敗：{e}")

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} JSON 頂層必須是 dict（key=檔名）")

    meta = {
        "model": model_name or path.stem,
        "time_str": time_str or "NA",
        "time_seconds": time_seconds if time_seconds is not None else -1,
    }
    return meta, data


@dataclass
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
    missing_pred_keys: int  # 答案有但預測沒有的 case 數
    extra_pred_keys: int    # 預測有但答案沒有的 case 數


def evaluate_one_model(
    gt: Dict[str, Dict[str, Any]],
    pred: Dict[str, Dict[str, Any]],
    *,
    fields: List[str],
) -> Tuple[int, int, float, Dict[str, float], int, int]:
    """
    回傳：
      total, strict_correct, field_acc_avg, per_field_acc, missing_pred, extra_pred
    """
    gt_keys = set(gt.keys())
    pred_keys = set(pred.keys())

    missing_pred = len(gt_keys - pred_keys)
    extra_pred = len(pred_keys - gt_keys)

    keys = sorted(gt_keys)
    total = len(keys)
    if total == 0:
        raise ValueError("答案檔沒有任何 case")

    strict_correct = 0
    field_hits_total = 0
    field_total = total * len(fields)

    per_field_hits = {f: 0 for f in fields}

    for k in keys:
        gt_obj = gt.get(k, {})
        pred_obj = pred.get(k, {})  # missing -> {}

        hits = 0
        for f in fields:
            gt_v = normalize_str(gt_obj.get(f))
            pred_v = normalize_str(pred_obj.get(f))
            ok = (gt_v == pred_v)
            if ok:
                hits += 1
                per_field_hits[f] += 1

        field_hits_total += hits
        if hits == len(fields):
            strict_correct += 1

    strict_acc = strict_correct / total
    field_acc_avg = field_hits_total / field_total
    per_field_acc = {f: per_field_hits[f] / total for f in fields}

    return total, strict_correct, strict_acc, field_acc_avg, per_field_acc, missing_pred, extra_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="答案檔 JSON 路徑（dict keyed-by-filename）")
    parser.add_argument("--pred_dir", required=True, help="模型輸出檔資料夾（多個 .json；含前四行 meta）")
    parser.add_argument("--out", default="report.json", help="輸出報表 JSON 檔名")
    args = parser.parse_args()

    answer_path = Path(args.answer)
    pred_dir = Path(args.pred_dir)
    out_path = Path(args.out)

    gt = load_json(answer_path)
    if not isinstance(gt, dict):
        raise ValueError("答案檔頂層必須是 dict（key=檔名）")

    reports: List[Dict[str, Any]] = []

    for p in sorted(pred_dir.glob("*.json")):
        meta, pred = parse_meta_and_json(p)

        total, strict_correct, strict_acc, field_acc_avg, per_field_acc, missing_pred, extra_pred = (
            evaluate_one_model(gt, pred, fields=FIELDS)
        )

        r = ModelReport(
            file=p.name,
            model=meta["model"],
            time_str=meta["time_str"],
            time_seconds=meta["time_seconds"],
            total=total,
            strict_correct=strict_correct,
            strict_accuracy=round(strict_acc, 6),
            field_accuracy=round(field_acc_avg, 6),
            per_field_accuracy={k: round(v, 6) for k, v in per_field_acc.items()},
            missing_pred_keys=missing_pred,
            extra_pred_keys=extra_pred,
        )

        reports.append(r.__dict__)

    # 排序：先 strict，再 field
    reports.sort(key=lambda x: (x["strict_accuracy"], x["field_accuracy"]), reverse=True)

    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_path.resolve()}")
    for r in reports:
        print(
            f"{r['model']:<20} | strict={r['strict_accuracy']:.3f} "
            f"| field={r['field_accuracy']:.3f} | time={r['time_str']} "
            f"| missing={r['missing_pred_keys']} extra={r['extra_pred_keys']}"
        )


if __name__ == "__main__":
    main()
