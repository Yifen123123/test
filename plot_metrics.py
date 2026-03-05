# plot_metrics.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="evaluator 輸出的 metrics.json")
    ap.add_argument("--out_dir", default="charts", help="圖表輸出資料夾")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics.json 應該是 list，且不能為空")

    # 依正確率排序（也可改成依時間或保持原順序）
    rows.sort(key=lambda r: r["acc"], reverse=True)

    models = [r["model"] for r in rows]
    accs = [r["acc"] for r in rows]
    secs = [r["time_seconds"] for r in rows]

    # 1) Accuracy chart
    plt.figure()
    plt.bar(models, accs)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=200)
    plt.close()

    # 2) Time chart
    plt.figure()
    plt.bar(models, secs)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time")
    plt.tight_layout()
    plt.savefig(out_dir / "time_seconds.png", dpi=200)
    plt.close()

    # 3) Optional: accuracy vs time scatter（看 trade-off）
    plt.figure()
    plt.scatter(secs, accs)
    for r in rows:
        plt.annotate(r["model"], (r["time_seconds"], r["acc"]))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_time.png", dpi=200)
    plt.close()

    print(f"Saved charts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
