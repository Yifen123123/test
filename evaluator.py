from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Config
# =========================
FIELDS = ["基準日", "來函機關", "收文編號", "查詢對象"]


# =========================
# Helpers
# =========================
def normalize_str(x: Any) -> str:
    """基本正規化：None -> '', 去頭尾空白、縮空白。"""
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_key(k: Any) -> str:
    """用於 case key（檔名）對齊：去頭尾空白、全形空白轉半形。"""
    s = str(k)
    s = s.replace("\u3000", " ")  # 全形空白
    return s.strip()


def _extract_json_text(text: str) -> str:
    i = text.find("{")
    if i == -1:
        raise ValueError("找不到 JSON 起點 '{'")
    return text[i:].strip()


def _clean_json_like(s: str) -> str:
    """
    容錯清理：
    - 智慧引號 -> "
    - 移除 //... 單行註解
    - 移除 /*...*/ 多行註解
    - 移除 trailing comma： ,} 或 ,]
    """
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    s = re.sub(r"//.*?$", "", s, flags=re.M)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)

    s = re.sub(r",\s*([}\]])", r"\1", s)

    return s.strip()


def _json_error_context(json_text: str, pos: int, window: int = 80) -> str:
    start = max(0, pos - window)
    end = min(len(json_text), pos + window)
    snippet = json_text[start:end]
    caret = " " * (pos - start) + "^"
    return snippet.replace("\n", "\\n") + "\n" + caret


# =========================
# Loaders
# =========================
def load_answer_json(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    答案檔建議是「純 JSON」：
    {
      "case_001.txt": {...},
      ...
    }
    若你答案檔也含 meta，你也可以改用 parse_prediction_file 來讀。
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("答案檔 JSON 頂層必須是 dict（key=檔名）")

    # key normalize
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        nk = normalize_key(k)
        out[nk] = v
    return out


def parse_prediction_file(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    預測檔格式（你貼的）：
    \"\"\" 
      資料：...
      模型：qwen2.5:3b
      總執行時間：3分10秒
    \"\"\"
    { ...JSON... }

    這裡不靠 regex 硬抓整段 meta，而是逐行掃描，最耐髒。
    """
    text = path.read_text(encoding="utf-8")

    model: Optional[str] = None
    time_str: Optional[str] = None
    time_seconds: Optional[int] = None

    def _norm_spaces(s: str) -> str:
        return s.replace("\u3000", " ").strip()

    for raw_line in text.splitlines():
        line = _norm_spaces(raw_line)

        if model is None and "模型：" in line:
            model = line.split("模型：", 1)[1].strip()

        if time_str is None and "總執行時間：" in line:
            tail = line.split("總執行時間：", 1)[1].strip()
            m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", tail)
            if m:
                minutes = int(m.group(1))
                seconds = int(m.group(2))
                time_seconds = minutes * 60 + seconds
                time_str = f"{minutes}分{seconds:02d}秒"
            else:
                # 若格式不符合（例如 190秒 或 00:12），先保留原字串
                time_str = tail

        if model is not None and time_str is not None:
            break

    json_text_raw = _extract_json_text(text)
    json_text = _clean_json_like(json_text_raw)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        ctx = _json_error_context(json_text, e.pos)
        raise ValueError(
            f"{path.name} JSON解析失敗：{e}\n"
            f"--- 錯誤附近上下文 ---\n{ctx}\n"
            f"（常見原因：key 沒雙引號、單引號、trailing comma、註解、智慧引號）"
        )

    if not isinstance(data, dict):
        raise ValueError(f"{path.name} JSON 頂層必須是 dict（key=檔名）")

    # key normalize
    pred: Dict[str, Dict[str, Any]] = {}
    for k, v in data.items():
        nk = normalize_key(k)
        pred[nk] = v

    meta = {
        "model": model or path.stem,
        "time_str": time_str or "NA",
        "time_seconds": time_seconds if time_seconds is not None else -1,
    }
    return meta, pred


# =========================
# Scoring
# =========================
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
        pred_obj = pred.get(k, {})  # 缺少 case -> 空 dict

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


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="答案檔（純 JSON）")
    parser.add_argument("--pred_dir", required=True, help="預測檔資料夾（多個 .json，允許含 meta + JSON）")
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

    reports.sort(key=lambda x: (x["strict_accuracy"], x["field_accuracy"]), reverse=True)

    out_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_path.resolve()}")
    for r in reports:
        print(
            f"{r['model']:<22} | strict={r['strict_accuracy']:.3f} "
            f"| field={r['field_accuracy']:.3f} | time={r['time_str']} "
            f"| missing={r['missing_pred_keys']} extra={r['extra_pred_keys']}"
        )


if __name__ == "__main__":
    main()
