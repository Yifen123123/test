# plot_metrics_pretty.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


def parse_time_seconds(time_str: str) -> int:
    m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", time_str or "")
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def pareto_frontier(points: list[tuple[int, float, str]]):
    """
    points: [(time_seconds, acc, model), ...]
    Pareto (min time, max acc): time 越小越好、acc 越大越好
    """
    pts = sorted(points, key=lambda x: (x[0], -x[1]))
    front = []
    best_acc = -1.0
    for t, acc, name in pts:
        if acc > best_acc:
            front.append((t, acc, name))
            best_acc = acc
    return front


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_dir", default="charts_pretty")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics.json 必須是非空 list")

    # 補齊 time_seconds
    for r in rows:
        if "time_seconds" not in r:
            r["time_seconds"] = parse_time_seconds(r.get("time_str", ""))

    # ========= 1) Scatter + Pareto =========
    points = [(int(r["time_seconds"]), float(r["acc"]), r["model"]) for r in rows]
    front = pareto_frontier(points)

    plt.figure(figsize=(8, 6))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    plt.scatter(xs, ys)

    # 標註點
    for t, acc, name in points:
        plt.annotate(name, (t, acc))

    # Pareto 線
    fx = [p[0] for p in front]
    fy = [p[1] for p in front]
    plt.plot(fx, fy)

    plt.xlabel("Time (seconds)  (lower is better)")
    plt.ylabel("Accuracy  (higher is better)")
    plt.title("Accuracy vs Time (Pareto frontier)")
    savefig(out_dir / "acc_vs_time_pareto.png")

    # ========= 2) Ranked Accuracy dot plot =========
    rows_acc = sorted(rows, key=lambda r: float(r["acc"]), reverse=True)
    labels = [r["model"] for r in rows_acc]
    accs = [float(r["acc"]) for r in rows_acc]
    y = list(range(len(labels)))

    plt.figure(figsize=(10, max(4, 0.45 * len(labels))))
    plt.scatter(accs, y)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlim(0, 1.0)
    plt.xlabel("Accuracy")
    plt.title("Ranked Accuracy")
    savefig(out_dir / "ranked_accuracy.png")

    # ========= 3) Ranked Time dot plot =========
    rows_t = sorted(rows, key=lambda r: int(r["time_seconds"]))
    labels = [r["model"] for r in rows_t]
    secs = [int(r["time_seconds"]) for r in rows_t]
    y = list(range(len(labels)))

    plt.figure(figsize=(10, max(4, 0.45 * len(labels))))
    plt.scatter(secs, y)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Time (seconds)")
    plt.title("Ranked Execution Time")
    savefig(out_dir / "ranked_time.png")

    print(f"Saved pretty charts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
