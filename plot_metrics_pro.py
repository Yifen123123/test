# plot_one.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def parse_time_seconds(time_str: str) -> int:
    m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", time_str or "")
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_family_and_size(model: str) -> Tuple[str, float]:
    """
    支援：
      qwen2.5:3b
      qwen2.5-7b
      llama3:8b
      phi-3-mini (抓不到 b 就 size=0)
    """
    s = (model or "").strip()
    low = s.lower()

    # 抓結尾的 "3b" / "7b" / "13b" / "1.7b" 等
    m = re.search(r"(.+?)[\s:_-]*([0-9]+(?:\.[0-9]+)?)b\b", low)
    if m:
        family = m.group(1).strip(" :_-")
        size = float(m.group(2))
        return family, size

    # 抓不到 b：整串當 family，size=0（會用固定深度）
    return low, 0.0


def shade_color(base_rgba, t: float):
    """
    t in [0,1]：越大越深
    用「混黑」方式：t 越大 -> 越靠近原色；t 越小 -> 越淡
    """
    r, g, b, a = base_rgba
    # 淡色比例：小模型更淡，大模型更接近原色
    # 0.35~1.00
    w = 0.35 + 0.65 * t
    return (r * w, g * w, b * w, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics.json", help="evaluator 產生的 metrics.json")
    ap.add_argument("--out", default="acc_vs_time.png", help="輸出圖片檔名")
    ap.add_argument("--title", default="Accuracy vs Execution Time", help="圖表標題")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics.json 必須是非空 list")

    # 準備資料
    models: List[str] = []
    accs: List[float] = []
    secs: List[int] = []

    families: List[str] = []
    sizes: List[float] = []

    for r in rows:
        model = str(r["model"])
        acc = float(r["acc"])
        sec = int(r.get("time_seconds") or parse_time_seconds(str(r.get("time_str", ""))))

        fam, sz = parse_family_and_size(model)

        models.append(model)
        accs.append(acc)
        secs.append(sec)
        families.append(fam)
        sizes.append(sz)

    # 讓顏色穩定：family 依字典序固定映射 tab10
    uniq_fams = sorted(set(families))
    fam_to_idx = {f: i for i, f in enumerate(uniq_fams)}
    base_cmap = cm.get_cmap("tab10")

    # 為了深淺合理：同 family 的 size 做 min-max normalize（若全是 0 就固定 0.7）
    fam_minmax: Dict[str, Tuple[float, float]] = {}
    for f in uniq_fams:
        ss = [s for fam, s in zip(families, sizes) if fam == f and s > 0]
        if ss:
            fam_minmax[f] = (min(ss), max(ss))
        else:
            fam_minmax[f] = (0.0, 0.0)

    colors = []
    for fam, sz in zip(families, sizes):
        base = base_cmap(fam_to_idx[fam] % 10)
        mn, mx = fam_minmax[fam]
        if mn == mx:
            t = 0.7 if sz == 0 else 1.0  # 沒大小資訊就中等深度；有但都一樣就最深
        else:
            t = (sz - mn) / (mx - mn)
        colors.append(shade_color(base, t))

    # 畫圖
    plt.figure(figsize=(9, 6))
    plt.scatter(secs, accs, c=colors, s=160, edgecolor="black", linewidths=0.6)

    # 標註（避免重疊：先用小偏移；你若點很多可再加進階避讓）
    for x, y, name in zip(secs, accs, models):
        plt.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points")

    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Legend：只放 family（顏色不帶深淺，代表色系）
    handles = []
    labels = []
    for fam in uniq_fams:
        base = base_cmap(fam_to_idx[fam] % 10)
        handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                 markerfacecolor=base, markeredgecolor="black",
                                 markersize=10))
        labels.append(fam)
    plt.legend(handles, labels, title="Model family", loc="best", frameon=True)

    plt.tight_layout()
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
