from pathlib import Path
from typing import List, Tuple, Iterable
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance, ImageFilter

# =========================
# 可調整參數
# =========================
INPUT_DIR   = Path("create_image_5")

OUT_BRIGHT  = Path("create_test_image/bright")
OUT_ROTATE  = Path("create_test_image/rotate")
OUT_NOISE   = Path("create_test_image/impurities")
OUT_DEGRADE = Path("create_test_image/degrade")

# 亮度（<1 變暗；>1 變亮）
BRIGHT_FACTORS: List[float] = [0.9, 0.8, 0.6, 1.1, 1.2, 1.4]

# 更貼近人為掃描的小歪斜（度）
ROTATE_DEGREES: List[float] = [-1.0, 0.8, 1.5]

# 旋轉背景色（避免四角出現黑邊）
ROTATE_BG = (255, 255, 255)

# 自然鹽椒雜訊：比例（salt:pepper）
NOISE_RATIOS: Iterable[Tuple[int, int]] = [(1, 3), (1, 1), (3, 1)]

# 雜訊密度：全圖中「種子點」比例（0.02=2%）
NOISE_DENSITY = 0.02

# 雜訊外觀控制（可微調讓更自然）
NOISE_BLUR_RADIUS_RANGE = (0.6, 1.8)      # 斑點模糊半徑（像素）
NOISE_ALPHA_THRESHOLD_RANGE = (0.2, 0.6)  # 斑點成形門檻
NOISE_ALPHA_STRENGTH = 1.0                # 0~1，整體透明度縮放

# 畫質降低參數（可多組一起輸出）
# 1) JPEG 壓縮品質：數值越低失真越重（1~95）
JPEG_QUALITIES: List[int] = [35, 25, 15]

# 2) 解析度降低：先縮小 scale 再放回原尺寸
DOWNSCALE_FACTORS: List[float] = [0.7, 0.5, 0.35]  # 0.35 ≈ 35%

# 3) 模糊強度（Gaussian blur 半徑）
BLUR_RADII: List[float] = [0.8, 1.2, 1.8]

# 若想結果可重現，設固定亂數種子（None 表示每次不同）
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

def _rand_uniform(a: float, b: float) -> float:
    return float(np.random.uniform(a, b))

def _generate_blob_alpha_mask(h: int, w: int, seeds_rc: np.ndarray,
                              blur_radius: float, threshold: float) -> np.ndarray:
    """
    由「種子點」生成柔邊小斑點的 alpha mask（0~1）。
    流程：二值種子 → 高斯模糊 → 門檻化 → 線性拉伸成柔邊 alpha。
    """
    base = np.zeros((h, w), dtype=np.uint8)
    if len(seeds_rc) > 0:
        rows, cols = seeds_rc[:, 0], seeds_rc[:, 1]
        base[rows, cols] = 255
    seed_img = Image.fromarray(base, mode="L")

    blurred = seed_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = np.asarray(blurred, dtype=np.float32) / 255.0

    thr = np.clip(threshold, 0.0, 0.95)
    alpha = (mask - thr) / max(1e-6, (1.0 - thr))
    alpha = np.clip(alpha, 0.0, 1.0)

    if NOISE_ALPHA_STRENGTH != 1.0:
        alpha *= float(NOISE_ALPHA_STRENGTH)

    return alpha  # HxW, float32, [0,1]

def add_salt_pepper_natural(img: Image.Image, density: float, salt_ratio: float) -> Image.Image:
    """
    以自然方式加入鹽椒：
    - 種子點（依 density/salt_ratio）
    - 高斯模糊形成斑點（柔邊 alpha）
    - 以 alpha 與白/黑做線性混合
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype(np.float32)
    h, w, _ = arr.shape
    total_pixels = h * w

    n_seeds = int(total_pixels * max(0.0, min(1.0, density)))
    if n_seeds <= 0:
        return rgb

    n_salt = int(n_seeds * salt_ratio)
    n_pepper = n_seeds - n_salt

    flat_idx = np.random.choice(total_pixels, size=n_seeds, replace=False)
    rows, cols = np.divmod(flat_idx, w)
    salt_rc = np.stack([rows[:n_salt], cols[:n_salt]], axis=1) if n_salt > 0 else np.zeros((0,2), int)
    pep_rc  = np.stack([rows[n_salt:], cols[n_salt:]], axis=1) if n_pepper > 0 else np.zeros((0,2), int)

    # 兩種斑點用不同隨機參數
    if len(salt_rc) > 0:
        br_s = _rand_uniform(*NOISE_BLUR_RADIUS_RANGE)
        th_s = _rand_uniform(*NOISE_ALPHA_THRESHOLD_RANGE)
        alpha_salt = _generate_blob_alpha_mask(h, w, salt_rc, br_s, th_s)
    else:
        alpha_salt = np.zeros((h, w), dtype=np.float32)

    if len(pep_rc) > 0:
        br_p = _rand_uniform(*NOISE_BLUR_RADIUS_RANGE)
        th_p = _rand_uniform(*NOISE_ALPHA_THRESHOLD_RANGE)
        alpha_pep = _generate_blob_alpha_mask(h, w, pep_rc, br_p, th_p)
    else:
        alpha_pep = np.zeros((h, w), dtype=np.float32)

    # 先加「椒」（暗），再加「鹽」（亮）
    if np.any(alpha_pep > 0):
        a = alpha_pep[..., None]
        arr = (1.0 - a) * arr + a * 0.0

    if np.any(alpha_salt > 0):
        a = alpha_salt[..., None]
        arr = (1.0 - a) * arr + a * 255.0

    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

# ---- 畫質降低（degrade）方法 ----
def degrade_jpeg_quality(img: Image.Image, quality: int = 30) -> Image.Image:
    """
    低品質 JPEG 再讀回，模擬壓縮失真。
    quality: 1~95（數字越低失真越多）
    """
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality), optimize=False)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def degrade_resolution(img: Image.Image, scale: float = 0.5) -> Image.Image:
    """
    解析度降低：先縮小再放大回原尺寸。
    """
    scale = float(scale)
    w, h = img.size
    w2 = max(1, int(w * scale))
    h2 = max(1, int(h * scale))
    small = img.resize((w2, h2), Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC)

def degrade_blur(img: Image.Image, radius: float = 1.2) -> Image.Image:
    """
    模糊：Gaussian blur 半徑愈大愈糊。
    """
    return img.filter(ImageFilter.GaussianBlur(radius))


# =========================
# 主流程
# =========================
def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)

    ensure_dirs(OUT_BRIGHT, OUT_ROTATE, OUT_NOISE, OUT_DEGRADE)

    files = sorted(INPUT_DIR.glob("*.png"))
    if not files:
        print(f"⚠️ 找不到 PNG：{INPUT_DIR} 下沒有 .png 檔。")
        return

    print(f"共找到 {len(files)} 張 PNG，開始處理…")

    for p in files:
        stem = p.stem
        img = Image.open(p).convert("RGB")

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

        # 3) 自然鹽椒雜訊
        for salt, pepper in NOISE_RATIOS:
            total = salt + pepper
            salt_ratio = salt / total
            out = add_salt_pepper_natural(img, density=NOISE_DENSITY, salt_ratio=salt_ratio)
            save_png(out, OUT_NOISE, stem, f"_sp{salt}-{pepper}_nat_d{int(NOISE_DENSITY*100)}p")

        # 4) 畫質降低（各種方式）
        #    (a) JPEG 壓縮
        for q in JPEG_QUALITIES:
            out = degrade_jpeg_quality(img, quality=q)
            save_png(out, OUT_DEGRADE, stem, f"_jpegQ{q}")

        #    (b) 解析度下降
        for s in DOWNSCALE_FACTORS:
            out = degrade_resolution(img, scale=s)
            save_png(out, OUT_DEGRADE, stem, f"_res{int(s*100)}")

        #    (c) 模糊
        for r in BLUR_RADII:
            out = degrade_blur(img, radius=r)
            # 半徑用小數一位比較直覺
            save_png(out, OUT_DEGRADE, stem, f"_blur{r:.1f}")

    print("✅ 完成！")
    print(f"亮度輸出：   {OUT_BRIGHT.resolve()}")
    print(f"旋轉輸出：   {OUT_ROTATE.resolve()}")
    print(f"雜訊輸出：   {OUT_NOISE.resolve()}")
    print(f"畫質降低輸出：{OUT_DEGRADE.resolve()}")


if __name__ == "__main__":
    main()
