import os
import json
import ast
import re
from typing import List, Tuple, Any
from PIL import Image, ImageDraw, ImageFont


# ----------------------------
# 1) 讀 positions：支援
#    A) [[[x1,y1],[x2,y2]], [[x1,y1],[x2,y2]], ...]
#    B) [[[[x1,y1],[x2,y2]]], [[[x1,y1],[x2,y2]]], ...]  (多一層包起來)
# ----------------------------
def load_positions(position_path: str) -> List[Tuple[int, int, int, int]]:
    """
    回傳 boxes: [(x1,y1,x2,y2), ...]
    會自動把各種嵌套格式 normalize。
    """
    with open(position_path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    try:
        data = ast.literal_eval(raw)  # 比 json.loads 更適合你這種 list 字面量格式
    except Exception as e:
        raise ValueError(f"positions.txt 不是合法的 Python list 字面量：{e}\n內容：{raw[:200]}...")

    def is_point(p: Any) -> bool:
        return isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, (int, float)) for x in p)

    def is_box(b: Any) -> bool:
        # [[x1,y1],[x2,y2]]
        return isinstance(b, (list, tuple)) and len(b) == 2 and is_point(b[0]) and is_point(b[1])

    boxes: List[Tuple[int, int, int, int]] = []

    def walk(node: Any):
        if is_box(node):
            (x1, y1), (x2, y2) = node
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            # 保證 x1<x2, y1<y2
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
# 2) 排版：自動換行 / 自動縮字 / 保留段落
# ----------------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    # 公文項次：確保前面有換行，讓版面漂亮
    s = re.sub(r"\s*(ㄧ、)", r"\nㄧ、", s)
    s = re.sub(r"\s*(一、)", r"\n一、", s)
    s = re.sub(r"\s*(二、)", r"\n二、", s)
    s = re.sub(r"\s*(三、)", r"\n三、", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def wrap_by_pixel(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    lines: List[str] = []
    for para in text.split("\n"):
        if para == "":
            lines.append("")  # 空行保留
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

def fit_text_to_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    font_path: str,
    max_font_size: int = 30,
    min_font_size: int = 12,
    line_spacing_ratio: float = 0.25,
    padding: int = 6,
):
    x1, y1, x2, y2 = box
    max_w = max(1, (x2 - x1) - padding * 2)
    max_h = max(1, (y2 - y1) - padding * 2)

    text = normalize_text(text)

    for size in range(max_font_size, min_font_size - 1, -1):
        font = ImageFont.truetype(font_path, size=size)
        line_spacing = int(size * line_spacing_ratio)

        lines = wrap_by_pixel(draw, text, font, max_w)

        # 算總高度
        total_h = 0
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln if ln else "　", font=font)
            total_h += (bbox[3] - bbox[1]) + line_spacing
        total_h = max(0, total_h - line_spacing)

        if total_h <= max_h:
            return font, lines, line_spacing, padding

    # 最小字還塞不下：截斷 + …
    font = ImageFont.truetype(font_path, size=min_font_size)
    line_spacing = int(min_font_size * line_spacing_ratio)
    lines = wrap_by_pixel(draw, text, font, max_w)

    allowed = []
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
    fill=(0, 0, 0),
    valign: str = "top",  # top / center
    max_font_size: int = 30,
    min_font_size: int = 12,
    line_spacing_ratio: float = 0.25,
    padding: int = 6,
):
    font, lines, line_spacing, padding = fit_text_to_box(
        draw=draw,
        text=text,
        box=box,
        font_path=font_path,
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
# 3) 主要流程：讀 JSON → 逐筆填字 → 輸出圖片
# ----------------------------
def main():
    image_path = "official_documents/parameter/data_001_whiteout.png"
    position_path = "official_documents/parameter/data_001_positions.txt"
    json_path = "create_data/azure_off_doc_10.json"
    output_dir = "create_image_doc"

    # macOS 字體（你若已有專案字體路徑就換掉）
    font_path = "/System/Library/Fonts/STHeiti Medium.ttc"

    boxes = load_positions(position_path)

    with open(json_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    for idx, case in enumerate(all_cases, start=1):
        # 你這裡的順序必須跟 boxes 順序一致（這是你目前的設計）
        # ✅ 收文編號一定兩行：用 "\n" 串在同一個字串
        收文編號 = case.get("收文編號", "")
        if isinstance(收文編號, list):
            # 若你 JSON 存成 ["113.01.01","11304123"]
            收文編號 = "\n".join(map(str, 收文編號))
        elif "\n" not in str(收文編號).strip():
            # 若只給一行，這裡也幫你補成兩行（第二行空白）
            收文編號 = (str(收文編號).strip() + "\n").strip("\n") + "\n"

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

        try:
            img = Image.open(image_path).convert("RGBA")
            draw = ImageDraw.Draw(img)

            for box, text in zip(boxes, fake_text_list):
                # 針對不同欄位可調參數（你可自行微調）
                # 這裡用 box 的 index 粗略區分：最後一個通常是「說明」
                is_last = (box == boxes[-1])

                if is_last:
                    # 說明：字小一點 + 行距大一點
                    draw_text_in_box(
                        draw=draw,
                        text=str(text),
                        box=box,
                        font_path=font_path,
                        valign="top",
                        max_font_size=24,
                        line_spacing_ratio=0.30,
                        padding=6,
                    )
                else:
                    draw_text_in_box(
                        draw=draw,
                        text=str(text),
                        box=box,
                        font_path=font_path,
                        valign="top",
                        max_font_size=28,
                        line_spacing_ratio=0.25,
                        padding=6,
                    )

            out_path = os.path.join(output_dir, f"standard_1_{idx}.png")
            img.convert("RGB").save(out_path, "PNG")
            print(f"✅ 產生成功：{out_path}")

        except Exception as e:
            print(f"❌ 第 {idx} 筆資料處理失敗：{e}")


if __name__ == "__main__":
    main()
