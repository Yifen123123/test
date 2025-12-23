import os
import json
import ast
from typing import List, Tuple, Dict
from PIL import Image, ImageDraw, ImageFont


# =====================================================
# 1) 讀取 positions.txt
#    格式：[[[x1,y1],[x2,y2]], [[x1,y1],[x2,y2]], ...]
# =====================================================
def load_table_boxes(position_path: str) -> List[Tuple[int, int, int, int]]:
    with open(position_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    data = ast.literal_eval(raw)

    boxes = []
    for box in data:
        (x1, y1), (x2, y2) = box
        x1, x2 = sorted((int(x1), int(x2)))
        y1, y2 = sorted((int(y1), int(y2)))
        boxes.append((x1, y1, x2, y2))

    if not boxes:
        raise ValueError("positions.txt 沒有任何 box")

    return boxes


# =====================================================
# 2) 字型載入
# =====================================================
def load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


# =====================================================
# 3) 只做「換行」，不縮字
# =====================================================
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


# =====================================================
# 4) 表格專用：固定字體大小畫字
# =====================================================
def draw_text_in_table_cell(
    draw,
    text,
    box,
    font_path,
    font_size=18,          # ⭐ 固定字級
    halign="left",         # left / center / right
    valign="center",       # center / top
    padding=4,
    fill=(0, 0, 0),
):
    font = load_font(font_path, font_size)

    x1, y1, x2, y2 = box
    max_w = (x2 - x1) - padding * 2
    max_h = (y2 - y1) - padding * 2

    lines = wrap_plain_by_pixel(draw, text, font, max_w)
    line_spacing = int(font_size * 0.2)

    # 計算高度，超出就截斷
    heights = [draw.textbbox((0, 0), ln, font=font)[3] for ln in lines]
    total_h = sum(heights) + line_spacing * (len(lines) - 1)

    while total_h > max_h and lines:
        lines.pop()
        heights.pop()
        total_h = sum(heights) + line_spacing * (len(lines) - 1)

    # 垂直起點
    if valign == "center":
        y = y1 + padding + max(0, (max_h - total_h) // 2)
    else:
        y = y1 + padding

    # 逐行畫
    for ln, h in zip(lines, heights):
        w = draw.textbbox((0, 0), ln, font=font)[2]

        if halign == "left":
            x = x1 + padding
        elif halign == "center":
            x = x1 + padding + max(0, (max_w - w) // 2)
        else:  # right
            x = x2 - padding - w

        draw.text((x, y), ln, font=font, fill=fill)
        y += h + line_spacing


# =====================================================
# 5) 攤平成表格順序（5 人 × 3 欄）
# =====================================================
def flatten_investigators_for_table(investigators: List[Dict[str, str]]) -> List[str]:
    result = []
    for p in investigators:
        result.append(p.get("姓名", ""))
        result.append(p.get("身分證字號", ""))
        result.append(p.get("財產基準日", ""))
    return result


# =====================================================
# 6) 主流程：讀 JSON → 每案輸出一張圖
# =====================================================
def main():
    image_path = "official_documents/parameter/data_001_whiteout.png"
    position_path = "official_documents/parameter/investigators_table_positions.txt"
    json_path = "investigators.json"
    output_dir = "create_image_doc"

    font_path = r"C:\Windows\Fonts\kaiu.ttf"  # Windows 標楷體
    font_size = 18                            # ⭐ 表格統一字級

    os.makedirs(output_dir, exist_ok=True)

    table_boxes = load_table_boxes(position_path)

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    for idx, case in enumerate(cases, start=1):
        investigators = case["調查人資料"]
        texts = flatten_investigators_for_table(investigators)

        if len(texts) != len(table_boxes):
            raise ValueError("表格框數量與調查人資料數量不一致")

        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)

        for i, (box, text) in enumerate(zip(table_boxes, texts)):
            col = i % 3
            halign = ["left", "center", "right"][col]

            draw_text_in_table_cell(
                draw=draw,
                text=text,
                box=box,
                font_path=font_path,
                font_size=font_size,   # ⭐ 全表格一致
                halign=halign,
                valign="center",
                padding=4,
            )

        out_path = os.path.join(output_dir, f"case_{idx}.png")
        img.convert("RGB").save(out_path)
        print(f"✅ 輸出完成：{out_path}")


if __name__ == "__main__":
    main()
