import os
import json
import ast
import re
from typing import List, Tuple, Any, Dict
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# 1) 讀 positions：支援你的格式
#    [[[x1,y1],[x2,y2]], [[x1,y1],[x2,y2]], ...]
#    或多一層包起來也會自動扁平化
# ----------------------------
def load_positions(position_path: str) -> List[Tuple[int, int, int, int]]:
    with open(position_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = ast.literal_eval(raw)
    except Exception as e:
        raise ValueError(f"positions.txt 不是合法的 Python list 字面量：{e}\n內容前200字：{raw[:200]}")

    def is_point(p: Any) -> bool:
        return isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, (int, float)) for x in p)

    def is_box(b: Any) -> bool:
        return isinstance(b, (list, tuple)) and len(b) == 2 and is_point(b[0]) and is_point(b[1])

    boxes: List[Tuple[int, int, int, int]] = []

    def walk(node: Any):
        if is_box(node):
            (x1, y1), (x2, y2) = node
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            boxes.append((x1, y1, x2, y2))
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(data)

    if not boxes:
        raise ValueError("positions.txt 解析後沒有找到任何 box（[[x1,y1],[x2,y2]]）")

    return boxes


# ----------------------------
# 2) 字型載入（避免 cannot open resource 直接炸）
# ----------------------------
def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(font_path, size=size)
    except OSError as e:
        raise OSError(
            f"cannot open resource：字型載入失敗。\n"
            f"font_path={font_path!r}\n"
            f"請確認檔案存在/權限正確，或改成專案內 fonts/*.ttf/.otf。\n"
            f"原始錯誤：{e}"
        )


# ----------------------------
# 3) 一般文字 normalize
# ----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    # 確保項次前換行（提高模型資料雜亂時的穩定度）
    s = re.sub(r"\s*(ㄧ、|一、|二、|三、|四、|五、|六、|七、|八、|九、|十、)", r"\n\1", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ----------------------------
# 4) 換行工具：一般版（像素）
# ----------------------------
def wrap_plain_by_pixel(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines: List[str] = []
    for para in text.split("\n"):
        if para == "":
            lines.append("")
            continue
        cur = ""
        for ch in para:
            test = cur + ch
            if draw.textlength(test, font=font) <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


# ----------------------------
# 5) 你要的：說明欄位懸掛縮排換行
# ----------------------------
_ITEM_RE = re.compile(r"^(ㄧ、|一、|二、|三、|四、|五、|六、|七、|八、|九、|十、)")

def wrap_hanging_items(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """
    目標：
    一、ABCDABCD
       ABCDABCD
    二、ABCDABCD
       ABCDABCD
    """
    text = normalize_text(text)
    out: List[str] = []

    for raw_line in text.split("\n"):
        raw_line = raw_line.strip()
        if not raw_line:
            out.append("")
            continue

        m = _ITEM_RE.match(raw_line)
        if not m:
            # 不是項次：照一般換行
            out.extend(wrap_plain_by_pixel(draw, raw_line, font, max_width))
            continue

        prefix = m.group(1)  # "一、"
        content = raw_line[len(prefix):].lstrip()

        # prefix 寬度（像素）→ 用空白逼近，確保對齊點準
        prefix_w = draw.textlength(prefix, font=font)
        indent = ""
        while draw.textlength(indent, font=font) < prefix_w:
            indent += " "

        # 第一行可用寬度要扣掉 prefix 寬
        first_max_w = max(1, max_width - int(prefix_w))

        wrapped = wrap_plain_by_pixel(draw, content, font, first_max_w)
        if not wrapped:
            out.append(prefix)
            continue

        out.append(prefix + wrapped[0])
        for ln in wrapped[1:]:
            out.append(indent + ln)

    return out


# ----------------------------
# 6) 自動縮字塞進框：支援 plain / hanging
# ----------------------------
def fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    font_path: str,
    mode: str = "plain",              # "plain" or "hanging"
    max_font_size: int = 28,
    min_font_size: int = 12,
    line_spacing_ratio: float = 0.25,
    padding: int = 6,
):
    x1, y1, x2, y2 = box
    max_w = max(1, (x2 - x1) - padding * 2)
    max_h = max(1, (y2 - y1) - padding * 2)

    text = normalize_text(text)

    for size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_path, size=size)
        line_spacing = int(size * line_spacing_ratio)

        if mode == "hanging":
            lines = wrap_hanging_items(draw, text, font, max_w)
        else:
            lines = wrap_plain_by_pixel(draw, text, font, max_w)

        total_h = 0
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln if ln else "　", font=font)
            total_h += (bbox[3] - bbox[1]) + line_spacing
        total_h = max(0, total_h - line_spacing)

        if total_h <= max_h:
            return font, lines, line_spacing, padding

    # 最小字還塞不下：截斷 + …
    font = load_font(font_path, size=min_font_size)
    line_spacing = int(min_font_size * line_spacing_ratio)

    if mode == "hanging":
        lines = wrap_hanging_items(draw, text, font, max_w)
    else:
        lines = wrap_plain_by_pixel(draw, text, font, max_w)

    allowed: List[str] = []
    used_h = 0
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln if ln else "　", font=font)
        h = bbox[3] - bbox[1]
        if used_h + h <= max_h:
            allowed.append(ln)
            used_h += h + line_spacing
        else:
            break

    if allowed:
        last = allowed[-1]
        allowed[-1] = (last[:-1] + "…") if last else "…"

    return font, allowed, line_spacing, padding


def draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    font_path: str,
    mode: str = "plain",
    fill=(0, 0, 0),
    valign: str = "top",              # "top" or "center"
    max_font_size: int = 28,
    min_font_size: int = 12,
    line_spacing_ratio: float = 0.25,
    padding: int = 6,
):
    font, lines, line_spacing, padding = fit_text_to_box(
        draw=draw,
        text=str(text),
        box=box,
        font_path=font_path,
        mode=mode,
        max_font_size=max_font_size,
        min_font_size=min_font_size,
        line_spacing_ratio=line_spacing_ratio,
        padding=padding,
    )

    x1, y1, x2, y2 = box
    max_h = (y2 - y1) - padding * 2

    heights = []
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln if ln else "　", font=font)
        heights.append(bbox[3] - bbox[1])

    total_h = sum(heights) + line_spacing * (len(lines) - 1) if lines else 0

    x = x1 + padding
    if valign == "center":
        y = y1 + padding + max(0, (max_h - total_h) // 2)
    else:
        y = y1 + padding

    for ln, h in zip(lines, heights):
        draw.text((x, y), ln, font=font, fill=fill)
        y += h + line_spacing


# ----------------------------
# 7) 主流程：讀 json → 依序填框 → 輸出
# ----------------------------
def main():
    # ✅ 按你的專案改路徑
    image_path = "official_documents/parameter/data_001_whiteout.png"
    position_path = "official_documents/parameter/data_001_positions.txt"
    json_path = "create_data/azure_off_doc_10.json"
    output_dir = "create_image_doc"

    # ✅ 字型路徑：若你不是 macOS 或路徑不存在，請改成專案 fonts/*.ttf
    font_path = "/System/Library/Fonts/PingFangTC.ttc"

    boxes = load_positions(position_path)

    with open(json_path, "r", encoding="utf-8") as f:
        all_cases: List[Dict] = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for idx, case in enumerate(all_cases, start=1):
        # ✅ 收文編號一定兩行：用 \n
        收文編號 = case.get("收文編號", "")
        if isinstance(收文編號, list):
            收文編號 = "\n".join(map(str, 收文編號))
        else:
            收文編號 = str(收文編號).strip()

        if "\n" not in 收文編號:
            # 若只有一行，第二行補空（你也可改成 raise 強制資料要兩行）
            收文編號 = 收文編號 + "\n"

        fake_text_list = [
            收文編號,
            case.get("發文者資料", ""),
            case.get("受文者資料", ""),
            case.get("主旨", ""),
            case.get("說明", ""),
        ]

        if len(fake_text_list) != len(boxes):
            raise ValueError(
                f"欄位數量不匹配：fake_text_list={len(fake_text_list)} 但 positions boxes={len(boxes)}。\n"
                f"你需要讓 positions 的框數量 = 你要填的欄位數量，且順序一致。"
            )

        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        for i, (box, text) in enumerate(zip(boxes, fake_text_list)):
            # 依序：0收文編號 1發文者 2受文者 3主旨 4說明
            if i == 4:
                # ✅ 說明用懸掛縮排（hanging indent）
                draw_text_in_box(
                    draw=draw,
                    text=str(text),
                    box=box,
                    font_path=font_path,
                    mode="hanging",
                    valign="top",
                    max_font_size=24,
                    line_spacing_ratio=0.30,
                    padding=6,
                )
            elif i == 3:
                # 主旨：稍微大字，行距小點
                draw_text_in_box(
                    draw=draw,
                    text=str(text),
                    box=box,
                    font_path=font_path,
                    mode="plain",
                    valign="top",
                    max_font_size=26,
                    line_spacing_ratio=0.22,
                    padding=6,
                )
            else:
                draw_text_in_box(
                    draw=draw,
                    text=str(text),
                    box=box,
                    font_path=font_path,
                    mode="plain",
                    valign="top",
                    max_font_size=26,
                    line_spacing_ratio=0.25,
                    padding=6,
                )

        out_path = os.path.join(output_dir, f"standard_1_{idx}.png")
        img.convert("RGB").save(out_path, "PNG")
        print(f"✅ 產生成功：{out_path}")


if __name__ == "__main__":
    main()
