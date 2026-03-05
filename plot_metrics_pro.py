# plot_metrics_single.py

import json
import re
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def parse_time_seconds(time_str):
    m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", time_str)
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--out", default="model_comparison.png")
    args = parser.parse_args()

    rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    models = []
    accs = []
    times = []

    for r in rows:
        models.append(r["model"])
        accs.append(r["acc"])
        times.append(parse_time_seconds(r["time_str"]))

    plt.figure(figsize=(8,6))

    scatter = plt.scatter(times, accs, c=range(len(models)), cmap="tab10", s=120)

    # 標註模型名稱
    for i, model in enumerate(models):
        plt.annotate(model, (times[i], accs[i]), xytext=(5,5), textcoords="offset points")

    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Execution Time")

    plt.grid(True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
