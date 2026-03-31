from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =========================
# 路徑設定
# =========================
INPUT_DIR = Path("input_txt")
OUTPUT_DIR = Path("output_txt")
FAKE_CSV = Path("fake_profiles.csv")

# 如果想固定每次抽到相同假資料，可改成整數，例如 42
RANDOM_SEED: Optional[int] = 42


# =========================
# CSV 欄位名稱設定
# =========================
CSV_COLS = {
    "birth_date": "birth_date",
    "id_number": "id_number",
    "mobile": "mobile",
    "email": "email",
}


# =========================
# 資料結構
# =========================
@dataclass
class FakeProfile:
    birth_date: str
    id_number: str
    mobile: str
    email: str


@dataclass
class ExtractedPII:
    birth_date: Optional[str] = None
    id_number: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None


# =========================
# Regex
# =========================

# Gmail
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@gmail\.com", re.IGNORECASE)

# 台灣身分證字號
ID_RE = re.compile(r"[A-Za-z][12]\d{8}")

# 台灣手機（正常完整格式）
MOBILE_RE = re.compile(r"09\d{8}")

# 民國日期（STT 版本）
# 支援：
# 1. 民國87年2月24
# 2. 民國87年02月24
# 3. 87年2月24
# 4. 870224
ROC_DATE_RE = re.compile(
    r"(民國\d{2,3}年\d{1,2}月\d{1,2})"
    r"|(\d{2,3}年\d{1,2}月\d{1,2})"
    r"|(\d{6})"
)

# 僅移除這些中文標點與空白，避免破壞 email
NORMALIZE_REMOVE_CHARS_RE = re.compile(r"[，。、「」『』【】（）()\[\]\s]+")


# =========================
# 工具函式
# =========================
def normalize_text_for_detection(text: str) -> str:
    """
    用於抽取 R: 內容後的偵測文字：
    - 去除空白與部分中文標點
    - 保留 email/date 所需字元
    """
    return NORMALIZE_REMOVE_CHARS_RE.sub("", text)


def normalize_mobile(s: str) -> str:
    digits = re.sub(r"\D", "", s)
    if digits.startswith("886") and len(digits) >= 11:
        digits = "0" + digits[3:]
    if len(digits) >= 10 and digits.startswith("09"):
        return digits[:10]
    return s.strip()


def normalize_id(s: str) -> str:
    return s.strip().upper()


def parse_roc_date(date_str: str) -> Tuple[int, int, int]:
    """
    將民國日期解析成 (roc_year, month, day)

    支援：
    - 民國87年2月24
    - 民國87年02月24
    - 87年2月24
    - 870224
    """
    s = date_str.strip()

    # 民國87年2月24
    m = re.fullmatch(r"民國(\d{2,3})年(\d{1,2})月(\d{1,2})", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # 87年2月24
    m = re.fullmatch(r"(\d{2,3})年(\d{1,2})月(\d{1,2})", s)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    # 870224
    m = re.fullmatch(r"(\d{2,3})(\d{2})(\d{2})", s)
    if m:
        roc_y = int(m.group(1))
        month = int(m.group(2))
        day = int(m.group(3))
        return roc_y, month, day

    raise ValueError(f"不支援的民國日期格式: {date_str}")


def format_fake_birth_like_original(original: str, fake_birth_date: str) -> str:
    """
    讓 fake 生日盡量跟原始格式一致
    """
    roc_y, m, d = parse_roc_date(fake_birth_date)
    original = original.strip()

    # 民國87年2月24
    if original.startswith("民國") and "年" in original and "月" in original:
        original_m = re.fullmatch(r"民國(\d{2,3})年(\d{1,2})月(\d{1,2})", original)
        if original_m:
            orig_month_str = original_m.group(2)
            orig_day_str = original_m.group(3)

            month_str = f"{m:02d}" if len(orig_month_str) == 2 else str(m)
            day_str = f"{d:02d}" if len(orig_day_str) == 2 else str(d)

            return f"民國{roc_y}年{month_str}月{day_str}"

    # 87年2月24
    if "年" in original and "月" in original:
        original_m = re.fullmatch(r"(\d{2,3})年(\d{1,2})月(\d{1,2})", original)
        if original_m:
            orig_month_str = original_m.group(2)
            orig_day_str = original_m.group(3)

            month_str = f"{m:02d}" if len(orig_month_str) == 2 else str(m)
            day_str = f"{d:02d}" if len(orig_day_str) == 2 else str(d)

            return f"{roc_y}年{month_str}月{day_str}"

    # 870224
    if re.fullmatch(r"\d{6}", original):
        return f"{roc_y:02d}{m:02d}{d:02d}"

    # fallback
    return f"民國{roc_y}年{m}月{d}"


def load_fake_profiles(csv_path: Path) -> List[FakeProfile]:
    profiles: List[FakeProfile] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            birth_date = row.get(CSV_COLS["birth_date"], "").strip()
            id_number = normalize_id(row.get(CSV_COLS["id_number"], ""))
            mobile = normalize_mobile(row.get(CSV_COLS["mobile"], ""))
            email = row.get(CSV_COLS["email"], "").strip()

            if not birth_date and not id_number and not mobile and not email:
                continue

            profiles.append(
                FakeProfile(
                    birth_date=birth_date,
                    id_number=id_number,
                    mobile=mobile,
                    email=email,
                )
            )

    if not profiles:
        raise ValueError("fake_profiles.csv 沒有有效資料")

    return profiles


def extract_r_lines(text: str) -> List[str]:
    """
    只抽 R: 的內容
    """
    r_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("R:"):
            r_lines.append(line[2:].strip())
    return r_lines


def build_detection_texts(r_lines: List[str]) -> List[str]:
    """
    建立三種偵測文本：
    1. 單行
    2. 相鄰兩行合併
    3. 相鄰三行合併
    4. 全部合併
    """
    texts: List[str] = []

    # 單行
    for line in r_lines:
        normalized = normalize_text_for_detection(line)
        if normalized:
            texts.append(normalized)

    # 相鄰兩行
    for i in range(len(r_lines) - 1):
        combined = normalize_text_for_detection(r_lines[i] + r_lines[i + 1])
        if combined:
            texts.append(combined)

    # 相鄰三行
    for i in range(len(r_lines) - 2):
        combined = normalize_text_for_detection(r_lines[i] + r_lines[i + 1] + r_lines[i + 2])
        if combined:
            texts.append(combined)

    # 全部合併
    all_combined = normalize_text_for_detection("".join(r_lines))
    if all_combined:
        texts.append(all_combined)

    texts = sorted(set(texts), key=len, reverse=True)
    return texts


def extract_pii_from_r_texts(r_lines: List[str]) -> ExtractedPII:
    """
    只從 R: 內容中抽取真正的個資
    """
    candidates = build_detection_texts(r_lines)
    extracted = ExtractedPII()

    for text in candidates:
        if extracted.email is None:
            m = EMAIL_RE.search(text)
            if m:
                extracted.email = m.group(0)

        if extracted.mobile is None:
            m = MOBILE_RE.search(text)
            if m:
                extracted.mobile = m.group(0)

        if extracted.id_number is None:
            m = ID_RE.search(text)
            if m:
                extracted.id_number = m.group(0).upper()

        if extracted.birth_date is None:
            m = ROC_DATE_RE.search(text)
            if m:
                extracted.birth_date = m.group(0)

        if (
            extracted.email is not None
            and extracted.mobile is not None
            and extracted.id_number is not None
            and extracted.birth_date is not None
        ):
            break

    return extracted


def build_replace_map(extracted: ExtractedPII, fake: FakeProfile) -> Dict[str, str]:
    """
    只針對成功抽到的真實個資建立替換表
    """
    replace_map: Dict[str, str] = {}

    if extracted.email and fake.email:
        replace_map[extracted.email] = fake.email

    if extracted.mobile and fake.mobile:
        replace_map[extracted.mobile] = fake.mobile

    if extracted.id_number and fake.id_number:
        replace_map[extracted.id_number] = fake.id_number

    if extracted.birth_date and fake.birth_date:
        replace_map[extracted.birth_date] = format_fake_birth_like_original(
            extracted.birth_date,
            fake.birth_date,
        )

    return replace_map


def replace_all_in_text(text: str, replace_map: Dict[str, str]) -> str:
    """
    對整份 txt 做替換。
    先以較長字串優先，避免部分覆蓋。
    """
    sorted_items = sorted(replace_map.items(), key=lambda x: len(x[0]), reverse=True)

    new_text = text
    for real_value, fake_value in sorted_items:
        if real_value:
            new_text = new_text.replace(real_value, fake_value)
    return new_text


def anonymize_one_text(text: str, fake: FakeProfile) -> Tuple[str, ExtractedPII, Dict[str, str]]:
    r_lines = extract_r_lines(text)
    extracted = extract_pii_from_r_texts(r_lines)
    replace_map = build_replace_map(extracted, fake)
    anonymized_text = replace_all_in_text(text, replace_map)
    return anonymized_text, extracted, replace_map


def main() -> None:
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fake_profiles = load_fake_profiles(FAKE_CSV)
    txt_files = sorted(INPUT_DIR.glob("*.txt"))

    if not txt_files:
        print("找不到任何 .txt 檔案")
        return

    for txt_path in txt_files:
        text = txt_path.read_text(encoding="utf-8")
        fake = random.choice(fake_profiles)

        try:
            anonymized_text, extracted, replace_map = anonymize_one_text(text, fake)
        except Exception as e:
            print("=" * 60)
            print(f"檔案：{txt_path.name}")
            print(f"處理失敗：{e}")
            continue

        output_path = OUTPUT_DIR / txt_path.name
        output_path.write_text(anonymized_text, encoding="utf-8")

        print("=" * 60)
        print(f"檔案：{txt_path.name}")
        print(f"抽到的真資料：{extracted}")
        print(f"替換表：{replace_map}")
        print(f"輸出：{output_path}")

    print("=" * 60)
    print("全部處理完成")


if __name__ == "__main__":
    main()
