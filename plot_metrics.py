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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out_dir", default="charts")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    rows.sort(key=lambda r: r.get("acc", 0.0), reverse=True)

    models = [r["model"] for r in rows]
    accs = [r["acc"] for r in rows]
    secs = [r.get("time_seconds", parse_time_seconds(r.get("time_str", ""))) for r in rows]

    plt.figure()
    plt.bar(models, accs)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(models, secs)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time")
    plt.tight_layout()
    plt.savefig(out_dir / "time_seconds.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(secs, accs)
    for r in rows:
        x = r.get("time_seconds", parse_time_seconds(r.get("time_str", "")))
        y = r["acc"]
        plt.annotate(r["model"], (x, y))
    plt.xlabel("Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Time")
    plt.tight_layout()
    plt.savefig(out_dir / "acc_vs_time.png", dpi=200)
    plt.close()

    print(f"Saved charts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
