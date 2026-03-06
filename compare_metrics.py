"""
compare_metrics.py

功能：
1. 讀取兩份 metrics.json
2. 比較第二次相較第一次的 accuracy 與 avg_time_ms 差異
3. 輸出 comparison_result.json
4. 繪製比較圖表到 charts/ 資料夾

執行方式：
python compare_metrics.py metrics_001.json metrics_002.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get_metric(model_data: Dict[str, Any], key: str, default: float = math.nan) -> float:
    value = model_data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compare_metrics(metrics_1: Dict[str, Any], metrics_2: Dict[str, Any]) -> Dict[str, Any]:
    all_models = sorted(set(metrics_1.keys()) | set(metrics_2.keys()))
    result: Dict[str, Any] = {}

    for model in all_models:
        m1 = metrics_1.get(model, {})
        m2 = metrics_2.get(model, {})

        acc1 = safe_get_metric(m1, "accuracy")
        acc2 = safe_get_metric(m2, "accuracy")
        time1 = safe_get_metric(m1, "avg_time_ms")
        time2 = safe_get_metric(m2, "avg_time_ms")

        acc_diff = acc2 - acc1 if not (math.isnan(acc1) or math.isnan(acc2)) else math.nan
        time_diff = time2 - time1 if not (math.isnan(time1) or math.isnan(time2)) else math.nan

        if not math.isnan(time1) and time1 != 0 and not math.isnan(time2):
            time_diff_pct = (time_diff / time1) * 100
        else:
            time_diff_pct = math.nan

        result[model] = {
            "run1_accuracy": acc1,
            "run2_accuracy": acc2,
            "accuracy_diff": acc_diff,
            "run1_avg_time_ms": time1,
            "run2_avg_time_ms": time2,
            "time_diff_ms": time_diff,
            "time_diff_pct": time_diff_pct,
        }

    return result


def save_json(data: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_accuracy_comparison(comparison: Dict[str, Any], output_dir: Path) -> None:
    models = list(comparison.keys())
    run1 = [comparison[m]["run1_accuracy"] for m in models]
    run2 = [comparison[m]["run2_accuracy"] for m in models]

    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], run1, width=width, label="Run 1")
    plt.bar([i + width / 2 for i in x], run2, width=width, label="Run 2")

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison: Run 1 vs Run 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=200)
    plt.close()


def plot_time_comparison(comparison: Dict[str, Any], output_dir: Path) -> None:
    models = list(comparison.keys())
    run1 = [comparison[m]["run1_avg_time_ms"] for m in models]
    run2 = [comparison[m]["run2_avg_time_ms"] for m in models]

    x = list(range(len(models)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width / 2 for i in x], run1, width=width, label="Run 1")
    plt.bar([i + width / 2 for i in x], run2, width=width, label="Run 2")

    plt.xticks(x, models, rotation=30, ha="right")
    plt.ylabel("Average Time (ms)")
    plt.title("Average Time Comparison: Run 1 vs Run 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "time_comparison.png", dpi=200)
    plt.close()


def plot_accuracy_diff(comparison: Dict[str, Any], output_dir: Path) -> None:
    models = list(comparison.keys())
    diffs = [comparison[m]["accuracy_diff"] for m in models]

    plt.figure(figsize=(12, 6))
    plt.bar(models, diffs)
    plt.axhline(0, linewidth=1)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Accuracy Difference (Run2 - Run1)")
    plt.title("Accuracy Improvement / Drop")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_diff.png", dpi=200)
    plt.close()


def plot_scatter_tradeoff(comparison: Dict[str, Any], output_dir: Path) -> None:
    plt.figure(figsize=(10, 7))

    for model, data in comparison.items():
        x1 = data["run1_avg_time_ms"]
        y1 = data["run1_accuracy"]
        x2 = data["run2_avg_time_ms"]
        y2 = data["run2_accuracy"]

        if any(math.isnan(v) for v in [x1, y1, x2, y2]):
            continue

        plt.scatter(x1, y1, s=80, marker="o")
        plt.scatter(x2, y2, s=80, marker="^")

        plt.plot([x1, x2], [y1, y2], linewidth=1)

        plt.text(x1, y1, f"{model} (R1)", fontsize=8)
        plt.text(x2, y2, f"{model} (R2)", fontsize=8)

    plt.xlabel("Average Time (ms)")
    plt.ylabel("Accuracy")
    plt.title("Prompt Change Trade-off: Time vs Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_tradeoff.png", dpi=200)
    plt.close()


def print_summary(comparison: Dict[str, Any]) -> None:
    print("\n===== Comparison Summary =====")
    for model, data in comparison.items():
        acc_diff = data["accuracy_diff"]
        time_diff = data["time_diff_ms"]

        if math.isnan(acc_diff) or math.isnan(time_diff):
            print(f"{model}: 資料不完整，無法比較")
            continue

        acc_word = "提升" if acc_diff > 0 else "下降" if acc_diff < 0 else "不變"
        time_word = "變慢" if time_diff > 0 else "變快" if time_diff < 0 else "不變"

        print(
            f"{model}: "
            f"accuracy {acc_word} {acc_diff:+.4f}, "
            f"time {time_word} {time_diff:+.2f} ms"
        )


def main() -> None:
    if len(sys.argv) != 3:
        print("用法：python compare_metrics.py metrics_001.json metrics_002.json")
        sys.exit(1)

    path1 = Path(sys.argv[1])
    path2 = Path(sys.argv[2])

    metrics_1 = load_json(str(path1))
    metrics_2 = load_json(str(path2))

    comparison = compare_metrics(metrics_1, metrics_2)

    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)

    save_json(comparison, Path("comparison_result.json"))
    plot_accuracy_comparison(comparison, output_dir)
    plot_time_comparison(comparison, output_dir)
    plot_accuracy_diff(comparison, output_dir)
    plot_scatter_tradeoff(comparison, output_dir)

    print_summary(comparison)
    print("\n已輸出：")
    print("- comparison_result.json")
    print("- charts/accuracy_comparison.png")
    print("- charts/time_comparison.png")
    print("- charts/accuracy_diff.png")
    print("- charts/scatter_tradeoff.png")


if __name__ == "__main__":
    main()
