import os
import json
import ast
import re
from typing import List, Tuple, Dict, Literal
from PIL import Image, ImageDraw, ImageFont


# =========================
# 1) 讀 positions.txt
# =========================
def load_table_boxes(position_path: str) -> List[Tuple[int, int, int, int]]:
    with open(position_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    data = ast.literal_eval(raw)

    boxes = []
    for i, box in enumerate(data, start=1):
        (x1, y1), (x2, y2) = box
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        raise ValueError("positions.txt 沒有任何 box")

    return boxes


# =========================
# 2) 字型
# =========================
def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


# =========================
# 3) 換行
# =========================
def wrap_plain_by_pixel(draw, text, font, max_width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            if draw.textlength(cur + ch, font=font) <= max_width:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines


# =========================
# 4) 自動縮字
# =========================
def fit_text_to_box(draw, text, box, font_path,
                    max_font_size=22, min_font_size=12,
                    line_spacing_ratio=0.2, padding=4):

    x1, y1, x2, y2 = box
    max_w = (x2 - x1) - padding * 2
    max_h = (y2 - y1) - padding * 2

    for size in range(max_font_size, min_font_size - 1, -1):
        font = load_font(font_path, size)
        line_spacing = int(size * line_spacing_ratio)
        lines = wrap_plain_by_pixel(draw, text, font, max_w)

        total_h = sum(
            draw.textbbox((0, 0), ln, font=font)[3]
            for ln in lines
        ) + line_spacing * (len(lines) - 1)

        if total_h <= max_h:
            return font, lines, line_spacing, padding

    font = load_font(font_path, min_font_size)
    return font, [text], int(min_font_size * line_spacing_ratio), padding


# =========================
# 5) 畫文字（支援左右置中）
# =========================
def draw_text_in_box(draw, text, box, font_path,
                     halign="left", valign="center",
                     max_font_size=22, padding=4):

    font, lines, line_spacing, padding = fit_text_to_box(
        draw, text, box, font_path, max_font_size=max_font_size, padding=padding
    )

    x1, y1, x2, y2 = box
    content_w = (x2 - x1) - padding * 2
    content_h = (y2 - y1) - padding * 2

    heights = [draw.textbbox((0, 0), ln, font=font)[3] for ln in lines]
    widths = [draw.textbbox((0, 0), ln, font=font)[2] for ln in lines]
    total_h = sum(heights) + line_spacing * (len(lines) - 1)

    y = y1 + padding + (content_h - total_h) // 2

    for ln, h, w in zip(lines, heights, widths):
        if halign == "left":
            x = x1 + padding
        elif halign == "center":
            x = x1 + padding + (content_w - w) // 2
        else:
            x = x2 - padding - w

        draw.text((x, y), ln, font=font, fill=(0, 0, 0))
        y += h + line_spacing


# =========================
# 6) 攤平調查人資料
# =========================
def flatten_investigators_for_table(investigators: List[Dict[str, str]]) -> List[str]:
    result = []
    for p in investigators:
        result.append(p["姓名"])
        result.append(p["身分證字號"])
        result.append(p["財產基準日"])
    return result


# =========================
# 7) 主流程：讀 JSON → 每案輸出一張圖
# =========================
def main():
    image_path = "official_documents/parameter/data_001_whiteout.png"
    position_path = "official_documents/parameter/investigators_table_positions.txt"
    json_path = "investigators.json"
    output_dir = "create_image_doc"

    font_path = r"C:\Windows\Fonts\kaiu.ttf"

    os.makedirs(output_dir, exist_ok=True)

    table_boxes = load_table_boxes(position_path)

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for idx, case in enumerate(cases, start=1):
        investigators = case["調查人資料"]
        texts = flatten_investigators_for_table(investigators)

        if len(texts) != len(table_boxes):
            raise ValueError("調查人資料數量與表格框數量不一致")

        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        for i, (box, text) in enumerate(zip(table_boxes, texts)):
            col = i % 3
            halign = ["left", "center", "right"][col]

            draw_text_in_box(
                draw=draw,
                text=text,
                box=box,
                font_path=font_path,
                halign=halign,
                max_font_size=22,
                padding=4,
            )

        out_path = os.path.join(output_dir, f"case_{idx}.png")
        img.convert("RGB").save(out_path)
        print(f"✅ 輸出完成：{out_path}")


if __name__ == "__main__":
    main()
