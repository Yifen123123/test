# plot_one.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def to_halfwidth(s: str) -> str:
    """把全形英數符號轉半形，避免像 'Ｂ' 這種字型缺字。"""
    out = []
    for ch in s:
        code = ord(ch)
        # 全形空白
        if code == 0x3000:
            out.append(" ")
        # 全形 ASCII 範圍：！(FF01) ~ ～(FF5E)
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def parse_time_seconds(time_str: str) -> int:
    m = re.search(r"(\d+)\s*分\s*(\d+)\s*秒", time_str or "")
    if not m:
        return 0
    return int(m.group(1)) * 60 + int(m.group(2))


def parse_family_and_size(model: str) -> Tuple[str, float]:
    """
    支援：
      qwen2.5:3b / qwen2.5-7b / llama3:8b / ...
    抓不到 b 則 size=0，family=整串
    """
    s = (model or "").strip().lower()
    m = re.search(r"(.+?)[\s:_-]*([0-9]+(?:\.[0-9]+)?)b\b", s)
    if m:
        family = m.group(1).strip(" :_-") or "unknown"
        size = float(m.group(2))
        return family, size
    return (s or "unknown"), 0.0


def shade(base_rgba, t: float):
    """t in [0,1]：越大越深；用『混白』做淡色，視覺更舒服。"""
    r, g, b, a = base_rgba
    # 混白：t 小 -> 更淡；t 大 -> 更接近原色
    w = 1.0 - (0.65 * t)  # 0.35~1.0
    return (r * (1 - w) + 1 * w, g * (1 - w) + 1 * w, b * (1 - w) + 1 * w, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="metrics.json")
    ap.add_argument("--out", default="acc_vs_time.png")
    ap.add_argument("--title", default="Accuracy vs Execution Time")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("metrics.json 必須是非空 list")

    # 準備資料（同時做全形->半形）
    models: List[str] = []
    accs: List[float] = []
    secs: List[int] = []
    families: List[str] = []
    sizes: List[float] = []

    for r in rows:
        model_raw = str(r["model"])
        model = to_halfwidth(model_raw)  # ✅ 修正 _Ｂ 類型缺字
        acc = float(r["acc"])
        sec = int(r.get("time_seconds") or parse_time_seconds(str(r.get("time_str", ""))))

        fam, sz = parse_family_and_size(model)

        models.append(model)
        accs.append(acc)
        secs.append(sec)
        families.append(fam)
        sizes.append(sz)

    # family 顏色：固定、穩定
    uniq_fams = sorted(set(families))
    fam_to_idx = {f: i for i, f in enumerate(uniq_fams)}
    base_cmap = cm.get_cmap("tab10")

    # 同 family 的 size 做 min-max normalize（若沒有 size 資訊就固定 0.7）
    fam_minmax: Dict[str, Tuple[float, float]] = {}
    for f in uniq_fams:
        ss = [s for fam, s in zip(families, sizes) if fam == f and s > 0]
        fam_minmax[f] = (min(ss), max(ss)) if ss else (0.0, 0.0)

    colors = []
    for fam, sz in zip(families, sizes):
        base = base_cmap(fam_to_idx[fam] % 10)
        mn, mx = fam_minmax[fam]
        if mn == mx:
            t = 0.7 if sz == 0 else 1.0
        else:
            t = (sz - mn) / (mx - mn)
        colors.append(shade(base, t))

    # 畫圖（加寬，因為 legend 要放右側）
    plt.figure(figsize=(11, 6))
    plt.scatter(secs, accs, c=colors, s=160, edgecolor="black", linewidths=0.6)

    # 標註模型（用半形字串，避免缺字）
    for x, y, name in zip(secs, accs, models):
        plt.annotate(name, (x, y), xytext=(6, 6), textcoords="offset points")

    plt.xlabel("Execution Time (seconds)")
    plt.ylabel("Accuracy")
    plt.title(args.title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # ✅ legend 放右側 + 字體放大，避免擠掉某些 family
    handles = []
    labels = []
    for fam in uniq_fams:
        base = base_cmap(fam_to_idx[fam] % 10)
        handles.append(
            plt.Line2D(
                [0], [0],
                marker="o", linestyle="",
                markerfacecolor=base,
                markeredgecolor="black",
                markersize=10,
            )
        )
        labels.append(fam)

    plt.legend(
        handles, labels,
        title="Model family",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=12,
        title_fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
