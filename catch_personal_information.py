from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# =========================
# 設定區
# =========================
INPUT_DIR = Path("input_txt")
OUTPUT_DIR = Path("output_txt")
FAKE_CSV = Path("fake_profiles.csv")

# 假資料 csv 欄位名稱
CSV_COLUMNS = {
    "name": "name",
    "birth_date": "birth_date",
    "id_number": "id_number",
    "address": "address",
    "mobile": "mobile",
    "email": "email",
}


# =========================
# 資料結構
# =========================
@dataclass
class Segment:
    speaker: Optional[str]   # "L" / "R" / None
    content: str


@dataclass
class CompactPos:
    seg_idx: int
    content_offset: int


@dataclass
class MatchItem:
    start: int
    end: int
    kind: str
    value: str


# =========================
# 讀取 fake csv
# =========================
def load_fake_profiles(csv_path: Path) -> List[Dict[str, str]]:
    profiles: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles.append({
                "name": row.get(CSV_COLUMNS["name"], "").strip(),
                "birth_date": row.get(CSV_COLUMNS["birth_date"], "").strip(),
                "id_number": row.get(CSV_COLUMNS["id_number"], "").strip(),
                "address": row.get(CSV_COLUMNS["address"], "").strip(),
                "mobile": normalize_mobile(row.get(CSV_COLUMNS["mobile"], "").strip()),
                "email": row.get(CSV_COLUMNS["email"], "").strip(),
            })
    if not profiles:
        raise ValueError("fake csv 沒有資料")
    return profiles


# =========================
# 文字解析
# =========================
LINE_PATTERN = re.compile(r"^(L|R)\s*:\s?(.*)$")

def parse_segments(text: str) -> List[Segment]:
    segments: List[Segment] = []
    for line in text.splitlines():
        m = LINE_PATTERN.match(line)
        if m:
            speaker = m.group(1)
            content = m.group(2)
            segments.append(Segment(speaker=speaker, content=content))
        else:
            # 非標準行，也保留
            segments.append(Segment(speaker=None, content=line))
    return segments


def render_segments(segments: List[Segment]) -> str:
    lines = []
    for seg in segments:
        if seg.speaker in {"L", "R"}:
            lines.append(f"{seg.speaker}: {seg.content}")
        else:
            lines.append(seg.content)
    return "\n".join(lines)


# =========================
# compact text 建立
# 去掉空白，保留字元位置映射
# =========================
def build_compact_text_and_map(segments: List[Segment]) -> Tuple[str, List[CompactPos]]:
    chars: List[str] = []
    pos_map: List[CompactPos] = []

    for seg_idx, seg in enumerate(segments):
        for offset, ch in enumerate(seg.content):
            if ch.isspace():
                continue
            chars.append(ch)
            pos_map.append(CompactPos(seg_idx=seg_idx, content_offset=offset))

    return "".join(chars), pos_map


# =========================
# 正規化工具
# =========================
def normalize_mobile(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if digits.startswith("886") and len(digits) >= 11:
        digits = "0" + digits[3:]
    if len(digits) >= 10 and digits.startswith("09"):
        return digits[:10]
    return s


def normalize_id(s: str) -> str:
    return s.strip().upper()


# =========================
# 日期格式轉換
# 讓假生日盡量模仿原格式
# =========================
def parse_fake_birth_date(fake_date: str) -> Tuple[int, int, int]:
    """
    支援：
    1998-10-21
    1998/10/21
    1998.10.21
    民國87年10月21日
    87/10/21
    """
    fake_date = fake_date.strip()

    m = re.match(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$", fake_date)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.match(r"^民國(\d{2,3})年(\d{1,2})月(\d{1,2})日$", fake_date)
    if m:
        y = int(m.group(1)) + 1911
        return y, int(m.group(2)), int(m.group(3))

    m = re.match(r"^(\d{2,3})[-/.](\d{1,2})[-/.](\d{1,2})$", fake_date)
    if m:
        y = int(m.group(1)) + 1911
        return y, int(m.group(2)), int(m.group(3))

    raise ValueError(f"無法解析 fake birth date: {fake_date}")


def format_birth_like_original(original: str, fake_date: str) -> str:
    y, m, d = parse_fake_birth_date(fake_date)
    roc_y = y - 1911

    if "民國" in original and "年" in original and "月" in original:
        return f"民國{roc_y}年{m}月{d}日"

    if re.fullmatch(r"\d{2,3}/\d{1,2}/\d{1,2}", original):
        return f"{roc_y}/{m}/{d}"

    if re.fullmatch(r"\d{2,3}-\d{1,2}-\d{1,2}", original):
        return f"{roc_y}-{m}-{d}"

    if re.fullmatch(r"\d{2,3}\.\d{1,2}\.\d{1,2}", original):
        return f"{roc_y}.{m}.{d}"

    if re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", original):
        return f"{y}/{m:02d}/{d:02d}"

    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", original):
        return f"{y}-{m:02d}-{d:02d}"

    if re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}", original):
        return f"{y}.{m:02d}.{d:02d}"

    if "年" in original and "月" in original and "日" in original:
        return f"{y}年{m}月{d}日"

    return fake_date


# =========================
# PII regex
# =========================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@gmail\.com", re.IGNORECASE)
ID_RE = re.compile(r"[A-Za-z][12]\d{8}")
MOBILE_RE = re.compile(r"(?:\+886)?09\d{8}|(?:\+886)9\d{8}")
# 日期：民國 / 西元 / 分隔符
DATE_RE = re.compile(
    r"(?:民國\d{2,3}年\d{1,2}月\d{1,2}日)|"
    r"(?:\d{2,3}[/-]\d{1,2}[/-]\d{1,2})|"
    r"(?:\d{2,3}\.\d{1,2}\.\d{1,2})|"
    r"(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|"
    r"(?:\d{4}\.\d{1,2}\.\d{1,2})|"
    r"(?:\d{4}年\d{1,2}月\d{1,2}日)"
)

# 地址先做較保守版，避免亂抓
ADDRESS_RE = re.compile(
    r"(?:臺|台)?[A-Za-z\u4e00-\u9fff]{1,6}(?:市|縣)"
    r"[A-Za-z\u4e00-\u9fff]{1,6}(?:區|鄉|鎮|市)"
    r"[A-Za-z0-9\u4e00-\u9fff]{1,20}(?:路|街|大道|巷)"
    r"[A-Za-z0-9\u4e00-\u9fff\-之]{0,10}"
    r"(?:段[A-Za-z0-9\u4e00-\u9fff]{0,5})?"
    r"\d{1,4}號"
    r"(?:\d{1,3}樓)?"
)

# 姓名建議保守抓：有明確上下文再抓
NAME_CONTEXT_RE_LIST = [
    re.compile(r"(?:我是|我叫|姓名是|名字是|請問您是)([\u4e00-\u9fff]{2,4})"),
    re.compile(r"([\u4e00-\u9fff]{2,4})(?:先生|小姐|女士|太太)"),
]


# =========================
# 找 match
# =========================
def find_matches(compact_text: str) -> List[MatchItem]:
    items: List[MatchItem] = []

    for m in EMAIL_RE.finditer(compact_text):
        items.append(MatchItem(m.start(), m.end(), "email", m.group(0)))

    for m in ID_RE.finditer(compact_text):
        items.append(MatchItem(m.start(), m.end(), "id_number", m.group(0)))

    for m in MOBILE_RE.finditer(compact_text):
        items.append(MatchItem(m.start(), m.end(), "mobile", m.group(0)))

    for m in DATE_RE.finditer(compact_text):
        items.append(MatchItem(m.start(), m.end(), "birth_date", m.group(0)))

    for m in ADDRESS_RE.finditer(compact_text):
        items.append(MatchItem(m.start(), m.end(), "address", m.group(0)))

    # 姓名最後抓，避免過度誤判
    for name_re in NAME_CONTEXT_RE_LIST:
        for m in name_re.finditer(compact_text):
            # group(1) 才是姓名
            name = m.group(1)
            start = m.start(1)
            end = m.end(1)
            items.append(MatchItem(start, end, "name", name))

    return merge_non_overlapping_matches(items)


def merge_non_overlapping_matches(items: List[MatchItem]) -> List[MatchItem]:
    """
    先按長度與位置排序，盡量保留較長 match，避免重疊。
    """
    items = sorted(items, key=lambda x: (x.start, -(x.end - x.start)))
    result: List[MatchItem] = []

    occupied = [False] * (max((x.end for x in items), default=0) + 1)

    for item in items:
        if any(occupied[i] for i in range(item.start, item.end)):
            continue
        for i in range(item.start, item.end):
            occupied[i] = True
        result.append(item)

    # 替換時從後往前，避免 offset 混亂
    result.sort(key=lambda x: x.start, reverse=True)
    return result


# =========================
# 依種類取得替換值
# =========================
def get_replacement(kind: str, original: str, fake_profile: Dict[str, str]) -> str:
    if kind == "email":
        return fake_profile["email"]
    if kind == "mobile":
        return fake_profile["mobile"]
    if kind == "id_number":
        return normalize_id(fake_profile["id_number"])
    if kind == "birth_date":
        return format_birth_like_original(original, fake_profile["birth_date"])
    if kind == "address":
        return fake_profile["address"]
    if kind == "name":
        return fake_profile["name"]
    return original


# =========================
# 核心替換：把 compact match 回寫到 segments
# =========================
def replace_match_in_segments(
    segments: List[Segment],
    pos_map: List[CompactPos],
    match: MatchItem,
    replacement: str
) -> None:
    start_pos = pos_map[match.start]
    end_pos = pos_map[match.end - 1]

    start_seg_idx = start_pos.seg_idx
    end_seg_idx = end_pos.seg_idx
    start_offset = start_pos.content_offset
    end_offset = end_pos.content_offset

    if start_seg_idx == end_seg_idx:
        seg = segments[start_seg_idx]
        seg.content = (
            seg.content[:start_offset]
            + replacement
            + seg.content[end_offset + 1:]
        )
        return

    # 跨多個 segment
    first_seg = segments[start_seg_idx]
    last_seg = segments[end_seg_idx]

    prefix = first_seg.content[:start_offset]
    suffix = last_seg.content[end_offset + 1:]

    first_seg.content = prefix + replacement
    for i in range(start_seg_idx + 1, end_seg_idx):
        segments[i].content = ""
    last_seg.content = suffix


# =========================
# 單一檔案處理
# =========================
def anonymize_text(text: str, fake_profile: Dict[str, str]) -> str:
    segments = parse_segments(text)
    compact_text, pos_map = build_compact_text_and_map(segments)
    matches = find_matches(compact_text)

    for m in matches:
        original = compact_text[m.start:m.end]
        replacement = get_replacement(m.kind, original, fake_profile)
        replace_match_in_segments(segments, pos_map, m, replacement)

        # 每次替換後要重建，因為 segments 已經變了
        compact_text, pos_map = build_compact_text_and_map(segments)

    return render_segments(segments)


# =========================
# 批次處理資料夾
# =========================
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fake_profiles = load_fake_profiles(FAKE_CSV)

    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        print("找不到 txt 檔")
        return

    for idx, txt_path in enumerate(txt_files):
        fake_profile = fake_profiles[idx % len(fake_profiles)]

        original_text = txt_path.read_text(encoding="utf-8")
        anonymized_text = anonymize_text(original_text, fake_profile)

        out_path = OUTPUT_DIR / txt_path.name
        out_path.write_text(anonymized_text, encoding="utf-8")
        print(f"完成：{txt_path.name} -> {out_path.name}")

    print("全部處理完成")


if __name__ == "__main__":
    main()
