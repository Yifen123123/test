# plot_metrics.py
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


def save_hbar(title: str, ylabel: str, labels: list[str], values: list[float], out_path: Path):
    # 高度隨模型數量變化，避免擠成一團
    h = max(4, 0.45 * len(labels))
    plt.figure(figsize=(10, h))
    plt.barh(labels, values)
    plt.gca().invert_yaxis()  # 讓排名第一在最上面
    plt.xlabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_dir", default="charts")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("metrics.json 是空的")

    # 依 acc 排序（從高到低）
    rows.sort(key=lambda r: r.get("acc", 0.0), reverse=True)

    labels = [r["model"] for r in rows]
    accs = [float(r["acc"]) for r in rows]
    secs = [r.get("time_seconds", parse_time_seconds(r.get("time_str", ""))) for r in rows]

    save_hbar(
        title="Model Accuracy",
        ylabel="Accuracy",
        labels=labels,
        values=accs,
        out_path=out_dir / "accuracy.png",
    )

    save_hbar(
        title="Execution Time",
        ylabel="Time (seconds)",
        labels=labels,
        values=secs,
        out_path=out_dir / "time_seconds.png",
    )

    # trade-off scatter（可選，但通常很有用）
    plt.figure(figsize=(8, 6))
    plt.scatter(secs, accs)
    for r in rows:
        x = r.get("time_seconds", parse_time_seconds(r.get("time_str", "")))
        y = float(r["acc"])
        plt.annotate(r["model"], (x, y))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_time.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved charts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
