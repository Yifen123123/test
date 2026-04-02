import json
from pathlib import Path
import pandas as pd


def is_empty(value) -> bool:
    """判斷 Excel 儲存格是否為空。"""
    if pd.isna(value):
        return True
    if str(value).strip() == "":
        return True
    return False


def excel_c_d_to_json(
    excel_path: str,
    output_path: str,
    id_prefix: str = "B",
    doc_type: str | None = None,
    sheet_name: str | int = 0,
    skip_header_row: bool = False,
) -> None:
    """
    將 Excel 的 C 欄(text) 與 D 欄(category) 轉成 JSON。

    參數:
        excel_path: Excel 檔案路徑，例如 "business.xlsx"
        output_path: 輸出的 JSON 檔案路徑，例如 "business_canned.json"
        id_prefix: ID 前綴，例如 "B" 或 "A"
        doc_type: 可選，若提供則會加入每筆資料，例如 "業務會辦單"
        sheet_name: 工作表名稱或索引，預設讀第一個工作表
        skip_header_row: 若你的第一列是標題，設為 True
    """

    excel_file = Path(excel_path)
    if not excel_file.exists():
        raise FileNotFoundError(f"找不到 Excel 檔案: {excel_path}")

    # header=None 代表不要把第一列當欄名，直接用位置抓欄位
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)

    # 如果第一列是標題，就跳過
    if skip_header_row:
        df = df.iloc[1:].reset_index(drop=True)

    results = []
    serial_no = 1

    for row_idx, row in df.iterrows():
        # C 欄 = index 2, D 欄 = index 3
        # 若該列欄位不足，直接跳過
        if len(row) < 4:
            continue

        text_value = row.iloc[2]
        category_value = row.iloc[3]

        # 跳過空白資料
        if is_empty(text_value) and is_empty(category_value):
            continue

        # text 為必要欄位，沒有就跳過
        if is_empty(text_value):
            print(f"第 {row_idx + 1} 列略過：C 欄(text)為空")
            continue

        text = str(text_value).strip()
        category = "" if is_empty(category_value) else str(category_value).strip()

        item = {
            "id": f"{id_prefix}{serial_no:03d}",
            "category": category,
            "text": text,
        }

        if doc_type is not None:
            item["doc_type"] = doc_type

        results.append(item)
        serial_no += 1

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"轉換完成，共輸出 {len(results)} 筆資料")
    print(f"輸出檔案：{output_file.resolve()}")


if __name__ == "__main__":
    # ===== 範例 1：業務會辦單 =====
    excel_c_d_to_json(
        excel_path="business.xlsx",
        output_path="business_canned.json",
        id_prefix="B",
        doc_type="業務會辦單",
        sheet_name=0,          # 第一個工作表
        skip_header_row=False  # 如果第一列是欄位名稱，改成 True
    )

    # ===== 範例 2：行政會辦單 =====
    # excel_c_d_to_json(
    #     excel_path="admin.xlsx",
    #     output_path="admin_canned.json",
    #     id_prefix="A",
    #     doc_type="行政會辦單",
    #     sheet_name=0,
    #     skip_header_row=False
    # )
