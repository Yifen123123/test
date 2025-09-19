from pathlib import Path
from typing import List, Tuple, Iterable
import numpy as np
from PIL import Image, ImageEnhance

# =========================
# 可調整參數
# =========================
INPUT_DIR   = Path("create_image_5")

OUT_BRIGHT  = Path("create_test_image/bright")
OUT_ROTATE  = Path("create_test_image/rotate")
OUT_NOISE   = Path("create_test_image/impurities")

# 亮度調整（<1 變暗；>1 變亮）
# 例如只要變暗可改成 [0.9, 0.8, 0.6]；只要變亮改成 [1.1, 1.2, 1.4]
BRIGHT_FACTORS: List[float] = [0.9, 0.8, 0.6, 1.1, 1.2, 1.4]

# 更貼近人為掃描的小歪斜（度）
ROTATE_DEGREES: List[float] = [-1.0, 0.8, 1.5]

# 旋轉背景色（避免四角出現黑邊）
ROTATE_BG = (255, 255, 255)

# 鹽椒雜訊比例（salt:pepper）
NOISE_RATIOS: Iterable[Tuple[int, int]] = [(1, 3), (1, 1), (3, 1)]

# 雜訊密度（整張圖中有多少比例像素被改成鹽/椒；0.02 = 2%）
NOISE_DENSITY = 0.02

# 若想結果可重現，可設定亂數種子（None 表示每次不同）
RANDOM_SEED: int | None = None


# =========================
# 小工具
# =========================
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def save_png(img: Image.Image, out_dir: Path, stem: str, suffix: str) -> None:
    out_path = out_dir / f"{stem}{suffix}.png"
    img.save(out_path, format="PNG", compress_level=6)

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)

def rotate_human_skew(img: Image.Image, degrees: float) -> Image.Image:
    """
    小角度旋轉。優先使用 fillcolor（Pillow 8+），
    舊版則改走 RGBA 合成到白底，避免黑角。
    """
    base = img.convert("RGB")
    try:
        return base.rotate(degrees, expand=True, resample=Image.BICUBIC, fillcolor=ROTATE_BG)
    except TypeError:
        rgba = base.convert("RGBA").rotate(degrees, expand=True, resample=Image.BICUBIC)
        bg = Image.new("RGBA", rgba.size, ROTATE_BG + (255,))
        composed = Image.alpha_composite(bg, rgba)
        return composed.convert("RGB")

def add_salt_pepper(img: Image.Image, density: float, salt_ratio: float) -> Image.Image:
    """
    對 RGB 影像加入鹽椒雜訊。
    density:  0~1，總雜訊像素比例（越大越吵）
    salt_ratio: 在雜訊像素中，鹽（白）的比例；椒（黑）比例 = 1 - salt_ratio
    """
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    h, w, _ = arr.shape
    total_pixels = h * w
    n_noisy = int(total_pixels * density)
    if n_noisy <= 0:
        return rgb

    # 亂數索引
    flat_indices = np.random.choice(total_pixels, size=n_noisy, replace=False)
    n_salt = int(n_noisy * salt_ratio)
    salt_idx = flat_indices[:n_salt]
    pepper_idx = flat_indices[n_salt:]

    # 映射回 (row, col)
    salt_rows, salt_cols = np.divmod(salt_idx, w)
    pep_rows, pep_cols   = np.divmod(pepper_idx, w)

    # 寫入白/黑
    arr[salt_rows,  salt_cols] = [255, 255, 255]
    arr[pep_rows,   pep_cols]  = [0, 0, 0]

    return Image.fromarray(arr)


# =========================
# 主流程
# =========================
def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    ensure_dirs(OUT_BRIGHT, OUT_ROTATE, OUT_NOISE)

    files = sorted(INPUT_DIR.glob("*.png"))
    if not files:
        print(f"⚠️ 找不到 PNG：{INPUT_DIR} 下沒有 .png 檔。")
        return

    print(f"共找到 {len(files)} 張 PNG，開始處理…")

    for p in files:
        stem = p.stem
        img = Image.open(p)

        # 1) 亮度（同時輸出變暗與變亮版本）
        for f in BRIGHT_FACTORS:
            out = adjust_brightness(img, f)
            delta = int(round((f - 1.0) * 100))  # 例如 -20 / +20
            sign = "+" if delta > 0 else ""
            save_png(out, OUT_BRIGHT, stem, f"_b{sign}{delta}p")

        # 2) 微角度旋轉（人為掃描小歪斜）
        for deg in ROTATE_DEGREES:
            out = rotate_human_skew(img, deg)
            sign = "+" if deg >= 0 else ""
            save_png(out, OUT_ROTATE, stem, f"_r{sign}{deg:.1f}deg")

        # 3) 鹽椒雜訊（不同 salt:pepper 比例）
        for salt, pepper in NOISE_RATIOS:
            total = salt + pepper
            salt_ratio = salt / total
            out = add_salt_pepper(img, density=NOISE_DENSITY, salt_ratio=salt_ratio)
            # 檔名例：_sp1-3_d2p（1:3，密度 2%）
            save_png(
                out,
                OUT_NOISE,
                stem,
                f"_sp{salt}-{pepper}_d{int(NOISE_DENSITY*100)}p"
            )

    print("✅ 完成！")
    print(f"亮度輸出：   {OUT_BRIGHT.resolve()}")
    print(f"旋轉輸出：   {OUT_ROTATE.resolve()}")
    print(f"雜訊輸出：   {OUT_NOISE.resolve()}")


if __name__ == "__main__":
    main()
