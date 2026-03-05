# plot_one.py

import argparse
import json
import re
import random
from pathlib import Path

import matplotlib.pyplot as plt


def parse_time_seconds(time_str):
    m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", time_str or "")
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_family(model):
    """
    取得模型 family
    qwen2.5:3b -> qwen2.5
    llama3:8b -> llama3
    """
    m = re.match(r"(.+?)[\:\-_]\d+\.?\d*b", model.lower())
    if m:
        return m.group(1)
    return model.lower()


def random_bright_color():
    """產生亮色"""
    return (
        random.uniform(0.4, 1),
        random.uniform(0.4, 1),
        random.uniform(0.4, 1),
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="metrics.json")
    parser.add_argument("--out", default="model_compare.png")
    args = parser.parse_args()

    rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    models = []
    accs = []
    times = []
    families = []

    for r in rows:
        model = r["model"]

        models.append(model)
        accs.append(r["acc"])
        times.append(parse_time_seconds(r["time_str"]))

        families.append(parse_family(model))

    # family -> color
    family_color = {}

    for f in set(families):
        family_color[f] = random_bright_color()

    colors = [family_color[f] for f in families]

    plt.figure(figsize=(9, 6))

    plt.scatter(
        times,
        accs,
        c=colors,
        s=180,
        edgecolor="black"
    )

    for i, model in enumerate(models):
        plt.annotate(
            model,
            (times[i], accs[i]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Execution Time")

    plt.grid(True)

    # legend (family)
    handles = []
    labels = []

    for f, c in family_color.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=c,
                markeredgecolor="black",
                markersize=10
            )
        )
        labels.append(f)

    plt.legend(handles, labels, title="Model family")

    plt.tight_layout()

    plt.savefig(args.out, dpi=200)

    print("saved:", args.out)


if __name__ == "__main__":
    main()
