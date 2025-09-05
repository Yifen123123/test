# -*- coding: utf-8 -*-
# pick_jsons.py  ->  parsed/<類別>/*.<ext>  ->  <out-root>/<類別>/{selected,others}/*
import argparse
import json
import random
import shutil
from pathlib import Path
from datetime import datetime

def safe_place(src_path: Path, dest_dir: Path, move: bool):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src_path.name
    if dest.exists():  # 避免覆蓋：加上 _1, _2, ...
        i = 1
        stem, suf = src_path.stem, src_path.suffix
        while (dest_dir / f"{stem}_{i}{suf}").exists():
            i += 1
        dest = dest_dir / f"{stem}_{i}{suf}"
    if move:
        shutil.move(str(src_path), str(dest))
    else:
        shutil.copy2(str(src_path), str(dest))
    return dest

def collect_categories(src_root: Path, ext: str):
    """只掃描一層：src_root/<category>/*{ext}"""
    cats = {}
    for d in sorted(p for p in src_root.iterdir() if p.is_dir()):
        files = sorted(p for p in d.glob(f"*{ext}") if p.is_file())
        if files:
            cats[d.name] = files
    return cats

def parse_per_cat_inline(expr: str) -> dict:
    """
    解析 --per-cat 例如： '保單查詢=20,保單查詢＋註記=5,其他=8'
    注意：類別名稱中若有空白或＋號都可，等號右邊要整數
    """
    mapping = {}
    if not expr:
        return mapping
    parts = [p.strip() for p in expr.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"--per-cat 格式錯誤：'{p}' 缺少 '='")
        k, v = p.split("=", 1)
        k, v = k.strip(), v.strip()
        if not v.isdigit():
            raise ValueError(f"--per-cat 數量需為整數：'{p}'")
        mapping[k] = int(v)
    return mapping

def load_per_cat_file(path: str) -> dict:
    """
    從 JSON 檔載入 per-cat 設定，格式：
    { "保單查詢": 20, "保單查詢＋註記": 5, "...": 8 }
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--per-cat-file 檔案不存在：{p.resolve()}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--per-cat-file JSON 必須是物件（鍵為類別、值為整數數量）")
    norm = {}
    for k, v in data.items():
        if not isinstance(v, int):
            raise ValueError(f"--per-cat-file 中類別 '{k}' 的數量必須是整數，收到：{type(v)}")
        norm[str(k)] = v
    return norm

def main():
    ap = argparse.ArgumentParser(
        description="從 <src-root>/<類別>/*.<ext> 每類別抽取指定數量，輸出為 <out-root>/<類別>/{selected,others}"
    )
    ap.add_argument("--src-root", type=str, default="parsed",
                    help="輸入根目錄（結構：<src-root>/<類別>/*.<ext>）")
    ap.add_argument("--out-root", type=str, default=None,
                    help="輸出根目錄（預設：<src-root>/_picked_YYYYmmdd_HHMMSS）")
    ap.add_argument("--n", type=int, default=5, help="每類別抽取數量（做為預設值）")
    ap.add_argument("--ratio", type=float, default=None,
                    help="（可選）每類別抽樣比例 0~1，與 --n 擇一，若同時提供以 per-cat/ratio 覆蓋 n")
    ap.add_argument("--move", action="store_true",
                    help="搬移檔案（預設為複製）")
    ap.add_argument("--seed", type=int, default=None,
                    help="隨機種子（指定可重現抽樣）")
    ap.add_argument("--ext", type=str, default=".json",
                    help="要處理的副檔名（預設 .json）")
    # 新增：每類別不同數量
    ap.add_argument("--per-cat", type=str, default=None,
                    help="以逗號分隔的 '類別=數量' 列表，如：'保單查詢=20,保單查詢＋註記=5'")
    ap.add_argument("--per-cat-file", type=str, default=None,
                    help="JSON 檔路徑，格式：{\"保單查詢\":20, \"保單查詢＋註記\":5}")

    args = ap.parse_args()

    src_root = Path(args.src_root)
    if not src_root.exists():
        raise SystemExit(f"找不到輸入根目錄：{src_root.resolve()}")

    out_root = Path(args.out_root) if args.out_root else (src_root / f"_picked_{datetime.now():%Y%m%d_%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)

    # 讀取 per-category 設定
    per_cat_map = {}
    # 檔案設定（較低優先）
    if args.per_cat_file:
        per_cat_map.update(load_per_cat_file(args.per_cat_file))
    # 內嵌設定（較高優先）
    if args.per_cat:
        per_cat_map.update(parse_per_cat_inline(args.per_cat))

    cats = collect_categories(src_root, args.ext)
    if not cats:
        raise SystemExit(f"在 {src_root.resolve()} 下面找不到任何含有 {args.ext} 的類別資料夾")

    # 基本檢查 ratio
    if args.ratio is not None and not (0.0 < args.ratio <= 1.0):
        raise SystemExit("--ratio 必須在 (0, 1]")

    total_sel = total_oth = 0
    print(f"🚀 掃描來源：{src_root.resolve()}")
    print(f"📦 輸出位置：{out_root.resolve()}")
    print(f"參數：預設 N={args.n}；" +
          (f"比例 ratio={args.ratio}；" if args.ratio is not None else "") +
          (f"模式={'搬移' if args.move else '複製'}；" ) +
          f"副檔名={args.ext}")
    if per_cat_map:
        print(f"🧩 每類別自訂數量：{per_cat_map}")

    # optional: 寫出 manifest
    manifest_lines = ["category,filename,chosen,src_path,dest_path"]

    for cat, files in cats.items():
        # 決定此類別抽樣數 k 的邏輯
        if cat in per_cat_map:
            k = per_cat_map[cat]
        elif args.ratio is not None:
            # 至少取 1 以避免空 selected（可依需求改成允許 0）
            k = max(1, int(round(len(files) * args.ratio)))
        else:
            k = args.n

        k = max(0, min(k, len(files)))  # 邊界保護
        picked = set(random.sample(files, k=k))

        cat_root = out_root / cat
        sel_dir = cat_root / "selected"
        oth_dir = cat_root / "others"

        count_sel = count_oth = 0
        for p in files:
            is_sel = p in picked
            dest_dir = sel_dir if is_sel else oth_dir
            dest_path = safe_place(p, dest_dir, move=args.move)
            if is_sel:
                count_sel += 1
            else:
                count_oth += 1
            manifest_lines.append(f"{cat},{p.name},{'1' if is_sel else '0'},{p.resolve()},{dest_path.resolve()}")

        total_sel += count_sel
        total_oth += count_oth
        print(f"   📁 類別「{cat}」：selected {count_sel}、others {count_oth}  ->  {cat_root}")

    # 輸出 manifest.csv
    manifest_path = out_root / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print("——")
    print(f"✅ 全部完成：selected {total_sel} 檔、others {total_oth} 檔")
    print(f"📝 抽樣清單：{manifest_path}")
    print(f"🗂️ 結構示意：")
    print(f"{out_root}/")
    print(f"  <類別A>/")
    print(f"    selected/*{args.ext}")
    print(f"    others/*{args.ext}")
    print(f"  <類別B>/ ...")

if __name__ == "__main__":
    main()
