from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
from PIL import Image, ImageEnhance

# =========================
# 可調整參數（你可依需求修改）
# =========================
INPUT_DIR = Path("create_image_5")

# 輸出資料夾
OUT_BRIGHT = Path("create_test_image/bright")
OUT_ROTATE = Path("create_test_image/rotate")
OUT_NOISE  = Path("create_test_image/impurities")

# 亮度等級（1.0=原始，>1 變亮，<1 變暗）
BRIGHT_FACTORS = [1.10, 1.20, 1.40]  # 對應 +10%、+20%、+40%

# 旋轉角度（度）
ROTATE_DEGREES = [4, 6, 8]

# 雜訊比例（salt:pepper）
NOISE_RATIOS: Iterable[Tuple[int, int]] = [(1, 3), (1, 1), (3, 1)]

# 雜訊密度（整張圖中要被改成鹽/椒的像素比例，0.02 = 2%）
NOISE_DENSITY = 0.02

# 旋轉時的背景色（避免出現黑角）
ROTATE_BGCOLOR = (255, 255, 255)


# =========================
# 功能函式
# =========================
def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """使用 PIL 的 ImageEnhance.Brightness 調整亮度"""
    return ImageEnhance.Brightness(img).enhance(factor)

def rotate_image(img: Image.Image, degrees: float) -> Image.Image:
    """小角度旋轉，expand=True 避免裁切，fillcolor 指定背景"""
    return img.convert("RGBA").rotate(degrees, expand=True, fillcolor=ROTATE_BGCOLOR).convert("RGB")

def add_salt_pepper(img: Image.Image, density: float, salt_ratio: float) -> Image.Image:
    """
    對 RGB 影像加入鹽椒雜訊。
    density:  整體雜訊比例（0~1）
    salt_ratio: 在雜訊像素中，鹽的比例（0~1）；椒比例 = 1 - salt_ratio
    """
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    h, w, c = arr.shape
    total_pixels = h * w
    n_noisy = int(total_pixels * density)
    if n_noisy <= 0:
        return rgb

    n_salt = int(n_noisy * salt_ratio)
    n_pepper = n_noisy - n_salt

    # 亂數挑像素索引
    flat_indices = np.random.choice(total_pixels, size=n_noisy, replace=False)
    salt_idx = flat_indices[:n_salt]
    pepper_idx = flat_indices[n_salt:]

    # 映射回 (row, col)
    salt_rows, salt_cols = np.divmod(salt_idx, w)
    pep_rows, pep_cols = np.divmod(pepper_idx, w)

    # 套用鹽（白）與椒（黑）
    arr[salt_rows, salt_cols] = [255, 255, 255]
    arr[pep_rows,  pep_cols]  = [0, 0, 0]

    return Image.fromarray(arr)

def save_variant(img: Image.Image, out_dir: Path, stem: str, suffix: str) -> None:
    out_path = out_dir / f"{stem}{suffix}.png"
    img.save(out_path, format="PNG", compress_level=6)


# =========================
# 主流程
# =========================
def main():
    ensure_dirs(OUT_BRIGHT, OUT_ROTATE, OUT_NOISE)

    pngs = sorted(INPUT_DIR.glob("*.png"))
    if not pngs:
        print(f"⚠️ 找不到 PNG：{INPUT_DIR} 下沒有 .png 檔。")
        return

    print(f"共找到 {len(pngs)} 張 PNG，開始處理…")

    for p in pngs:
        # 以檔名（不含副檔名）作為基底
        stem = p.stem
        img = Image.open(p)

        # 1) 亮度
        for f in BRIGHT_FACTORS:
            out = adjust_brightness(img, f)
            delta_percent = int(round((f - 1.0) * 100))
            save_variant(out, OUT_BRIGHT, stem, suffix=f"_b{delta_percent}p")

        # 2) 旋轉
        for deg in ROTATE_DEGREES:
            out = rotate_image(img, deg)
            save_variant(out, OUT_ROTATE, stem, suffix=f"_r{deg}deg")

        # 3) 鹽椒雜訊
        for salt, pepper in NOISE_RATIOS:
            total = salt + pepper
            salt_ratio = salt / total
            out = add_salt_pepper(img, density=NOISE_DENSITY, salt_ratio=salt_ratio)
            # 例：_sp1-3_d2p (2% noise density)
            save_variant(
                out, OUT_NOISE, stem,
                suffix=f"_sp{salt}-{pepper}_d{int(NOISE_DENSITY*100)}p"
            )

    print("✅ 完成！")
    print(f"亮度輸出：   {OUT_BRIGHT.resolve()}")
    print(f"旋轉輸出：   {OUT_ROTATE.resolve()}")
    print(f"雜訊輸出：   {OUT_NOISE.resolve()}")


if __name__ == "__main__":
    main()
