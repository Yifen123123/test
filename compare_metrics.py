"""
compare_metrics.py

功能：
1. 讀取兩份 metrics.json（list 格式）
2. 比較第二次相較第一次的 acc 與 time_str 差異
3. 輸出 comparison_result.json
4. 繪製比較圖表到 charts/ 資料夾

支援的 metrics.json 格式範例：
[
  {
    "model": "qwen2.5:3b",
    "acc": 0.82,
    "time_str": "2分15秒"
  },
  {
    "model": "qwen2.5:7b",
    "acc": 0.78,
    "time_str": "1分05秒"
  }
]

執行方式：
python compare_metrics.py metrics_001.json metrics_002.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def load_json(path: str) -> Any:
    """讀取 JSON 檔案"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_to_dict(metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    將 list 格式轉成以 model 為 key 的 dict
    例如：
    [
      {"model": "qwen2.5:3b", "acc": 0.8, "time_str": "1分20秒"}
    ]
    轉成：
    {
      "qwen2.5:3b": {"model": ..., "acc": ..., "time_str": ...}
    }
    """
    result: Dict[str, Dict[str, Any]] = {}

    for item in metrics_list:
        if not isinstance(item, dict):
            continue

        model = item.get("model")
        if not model:
            continue

        result[str(model)] = item

    return result


def parse_time_str(time_str: Any) -> float:
    """
    將時間字串轉成秒數
    支援格式：
    - 2分15秒
    - 15秒
    - 3分
    - 125.5秒
    失敗則回傳 nan
    """
    if not isinstance(time_str, str):
        return math.nan

    time_str = time_str.strip()

    # 例如：2分15秒
    match_min_sec = re.fullmatch(r"(?:(\d+)\s*分)?\s*(?:(\d+(?:\.\d+)?)\s*秒)?", time_str)
    if not match_min_sec:
        return math.nan

    min_part = match_min_sec.group(1)
    sec_part = match_min_sec.group(2)

    if min_part is None and sec_part is None:
        return math.nan

    minutes = int(min_part) if min_part is not None else 0
    seconds = float(sec_part) if sec_part is not None else 0.0

    return minutes * 60 + seconds


def format_seconds_to_time_str(total_seconds: float) -> str:
    """把秒數轉回 人類較好讀的格式"""
    if math.isnan(total_seconds):
        return "N/A"

    if total_seconds < 0:
        sign = "-"
        total_seconds = abs(total_seconds)
    else:
        sign = ""

    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60

    if minutes > 0:
        if seconds.is_integer():
            return f"{sign}{minutes}分{int(seconds)}秒"
        return f"{sign}{minutes}分{seconds:.2f}秒"

    if seconds.is_integer():
        return f"{sign}{int(seconds)}秒"
    return f"{sign}{seconds:.2f}秒"


def safe_float(value: Any) -> float:
    """安全轉 float，失敗回傳 nan"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def compare_metrics(
    metrics_1: Dict[str, Dict[str, Any]],
    metrics_2: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """比較兩次 metrics"""
    all_models = sorted(set(metrics_1.keys()) | set(metrics_2.keys()))
    result: Dict[str, Dict[str, Any]] = {}

    for model in all_models:
        m1 = metrics_1.get(model, {})
        m2 = metrics_2.get(model, {})

        acc1 = safe_float(m1.get("acc"))
        acc2 = safe_float(m2.get("acc"))

        time_str1 = m1.get("time_str", "")
        time_str2 = m2.get("time_str", "")

        time1_sec = parse_time_str(time_str1)
        time2_sec = parse_time_str(time_str2)

        acc_diff = acc2 - acc1 if not (math.isnan(acc1) or math.isnan(acc2)) else math.nan
        time_diff_sec = (
            time2_sec - time1_sec
            if not (math.isnan(time1_sec) or math.isnan(time2_sec))
            else math.nan
        )

        if not math.isnan(time1_sec) and time1_sec != 0 and not math.isnan(time2_sec):
            time_diff_pct = (time_diff_sec / time1_sec) * 100
        else:
            time_diff_pct = math.nan

        result[model] = {
            "run1_acc": acc1,
            "run2_acc": acc2,
            "acc_diff": acc_diff,
            "run1_time_str": time_str1,
            "run2_time_str": time_str2,
            "run1_time_sec": time1_sec,
            "run2_time_sec": time2_sec,
            "time_diff_sec": time_diff_sec,
            "time_diff_pct": time_diff_pct,
        }

    return result


def save_json(data: Dict[str, Any], path: Path) -> None:
    """儲存 JSON"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_acc_comparison(comparison: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """畫 acc 比較圖"""
    models = list(comparison.keys())
    run1 = [comparison[m]["run1_acc"] for m in models]
    run2 = [comparison[m]["run2_acc"] for m in models]

    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], run1, width=width, label="Run 1")
    plt.bar([i + width / 2 for i in x], run2, width=width, label="Run 2")

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "acc_comparison.png", dpi=200)
    plt.close()


def plot_time_comparison(comparison: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """畫時間比較圖（秒）"""
    models = list(comparison.keys())
    run1 = [comparison[m]["run1_time_sec"] for m in models]
    run2 = [comparison[m]["run2_time_sec"] for m in models]

    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], run1, width=width, label="Run 1")
    plt.bar([i + width / 2 for i in x], run2, width=width, label="Run 2")

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=200)
    plt.close()


def plot_acc_diff(comparison: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """畫 acc 差值圖"""
    models = list(comparison.keys())
    diffs = [comparison[m]["acc_diff"] for m in models]

    plt.figure(figsize=(12, 6))
    plt.bar(models, diffs)
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accuracy Difference (Run2 - Run1)")
    plt.title("Accuracy Difference")
    plt.tight_layout()
    plt.savefig(output_dir / "acc_diff.png", dpi=200)
    plt.close()


def plot_scatter_tradeoff(comparison, output_dir):
    """
    畫 time vs acc 散點圖
    改善：
    - label 字體變大
    - label 加 offset
    - 避免貼在點上
    """

    plt.figure(figsize=(11, 7))

    for model, data in comparison.items():
        x1 = data["run1_time_sec"]
        y1 = data["run1_acc"]
        x2 = data["run2_time_sec"]
        y2 = data["run2_acc"]

        if any(math.isnan(v) for v in [x1, y1, x2, y2]):
            continue

        # 畫點
        plt.scatter(x1, y1, s=90, marker="o")
        plt.scatter(x2, y2, s=90, marker="^")

        # 畫連線
        plt.plot([x1, x2], [y1, y2], linewidth=1)

        # label offset
        offset_x = (x2 - x1) * 0.05 + 0.02
        offset_y = (y2 - y1) * 0.05 + 0.002

        # Run1 label
        plt.text(
            x1 - offset_x,
            y1 + offset_y,
            f"{model} R1",
            fontsize=10,
            ha="right"
        )

        # Run2 label
        plt.text(
            x2 + offset_x,
            y2 + offset_y,
            f"{model} R2",
            fontsize=10,
            ha="left"
        )

    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Prompt Change Trade-off: Time vs Accuracy", fontsize=14)

    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "scatter_tradeoff.png", dpi=220)
    plt.close()


def print_summary(comparison: Dict[str, Dict[str, Any]]) -> None:
    """在終端機印出摘要"""
    print("\n===== Comparison Summary =====")
    for model, data in comparison.items():
        acc_diff = data["acc_diff"]
        time_diff_sec = data["time_diff_sec"]

        if math.isnan(acc_diff) or math.isnan(time_diff_sec):
            print(f"{model}: 資料不完整，無法比較")
            continue

        acc_word = "提升" if acc_diff > 0 else "下降" if acc_diff < 0 else "不變"
        time_word = "變慢" if time_diff_sec > 0 else "變快" if time_diff_sec < 0 else "不變"

        print(
            f"{model}: "
            f"acc {acc_word} {acc_diff:+.4f}，"
            f"time {time_word} {format_seconds_to_time_str(time_diff_sec)}"
        )


def main() -> None:
    if len(sys.argv) != 3:
        print("用法：python compare_metrics.py metrics_001.json metrics_002.json")
        sys.exit(1)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])

    if not path1.exists():
        print(f"找不到檔案：{path1}")
        sys.exit(1)

    if not path2.exists():
        print(f"找不到檔案：{path2}")
        sys.exit(1)

    raw_1 = load_json(str(path1))
    raw_2 = load_json(str(path2))

    if not isinstance(raw_1, list):
        print(f"{path1} 格式錯誤：預期是 list")
        sys.exit(1)

    if not isinstance(raw_2, list):
        print(f"{path2} 格式錯誤：預期是 list")
        sys.exit(1)

    metrics_1 = list_to_dict(raw_1)
    metrics_2 = list_to_dict(raw_2)

    comparison = compare_metrics(metrics_1, metrics_2)

    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)

    save_json(comparison, Path("comparison_result.json"))
    plot_acc_comparison(comparison, output_dir)
    plot_time_comparison(comparison, output_dir)
    plot_acc_diff(comparison, output_dir)
    plot_scatter_tradeoff(comparison, output_dir)

    print_summary(comparison)
    print("\n已輸出：")
    print("- comparison_result.json")
    print("- charts/acc_comparison.png")
    print("- charts/time_comparison.png")
    print("- charts/acc_diff.png")
    print("- charts/scatter_tradeoff.png")


if __name__ == "__main__":
    main()
