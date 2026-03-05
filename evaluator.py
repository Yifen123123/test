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
    text = path.read_text(encoding="utf-8")

    # --- meta: 用更穩的方式抓（逐行掃描，容忍空白/縮排/全形空白） ---
    model = None
    time_str = None
    time_seconds = None

    # 把全形空白轉半形，避免你檔案裡縮排被 regex 擋掉
    def _norm_spaces(s: str) -> str:
        return s.replace("\u3000", " ").strip()  # \u3000 = 全形空白

    for raw_line in text.splitlines():
        line = _norm_spaces(raw_line)

        # 形如：模型：qwen2.5:3b
        if "模型：" in line and model is None:
            # 只切第一個 "模型："，避免後面還有冒號
            model = line.split("模型：", 1)[1].strip()

        # 形如：總執行時間：3分10秒
        if "總執行時間：" in line and time_str is None:
            tail = line.split("總執行時間：", 1)[1].strip()
            # 抓 3分10秒 / 03分10秒 都可
            m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", tail)
            if m:
                minutes = int(m.group(1))
                seconds = int(m.group(2))
                time_seconds = minutes * 60 + seconds
                time_str = f"{minutes}分{seconds:02d}秒"
            else:
                # 有些人會輸出 190秒 或 00:12 之類，你也可在這裡加規則
                time_str = tail

        if model is not None and time_str is not None:
            break

    # --- JSON 本體：從第一個 '{' 開始 ---
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError(f"{path.name} 找不到 JSON 起點 '{{'")

    json_text_raw = text[json_start:].strip()

    # 容錯清理：智慧引號/註解/多餘逗號
    json_text = _clean_json_like(json_text_raw)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        ctx = _json_error_context(json_text, e.pos)
        raise ValueError(
            f"{path.name} JSON解析失敗：{e}\n"
            f"--- 錯誤附近上下文 ---\n{ctx}"
        )

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} JSON 頂層必須是 dict（key=檔名）")

    meta = {
        "model": model or path.stem,               # ✅ 現在大概率能抓到模型名
        "time_str": time_str or "NA",             # ✅ 現在大概率能抓到時間
        "time_seconds": time_seconds if time_seconds is not None else -1,
        "pred_keys": len(data),                   # ✅ 加這個方便你檢查 missing/extra
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
