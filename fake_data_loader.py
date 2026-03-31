import csv
import random

FAKE_DATA_PATH = "./fake_data/fake_data.csv"  # ← 改成你的 CSV 路徑

# 對應 CSV 欄位名稱 → 程式內部使用的 category key
COLUMN_MAP = {
    "name":       "name",
    "birth_date": "birthday",
    "id_number":  "id",
    "address":    "address",
    "mobile":     "phone",
    "email":      "email",
}

# 啟動時載入一次
_pools: dict[str, list[str]] = {key: [] for key in COLUMN_MAP.values()}

def _load():
    with open(FAKE_DATA_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for csv_col, category in COLUMN_MAP.items():
                value = row.get(csv_col, "").strip()
                if value:
                    _pools[category].append(value)

    # 確認每個欄位都有資料
    for category, values in _pools.items():
        if not values:
            raise ValueError(f"假資料欄位「{category}」是空的，請確認 CSV 內容")

_load()

def get_fake(category: str) -> str:
    """隨機取一筆假資料，category 可為: name / birthday / id / address / phone / email"""
    if category not in _pools:
        raise KeyError(f"未知的 category：{category}，可用值為 {list(_pools.keys())}")
    return random.choice(_pools[category])
```

---

### CSV 格式確認

你的 CSV 只要長這樣就可以直接用：
```
name,birth_date,id_number,address,mobile,email
王大明,1985/03/15,A123456789,台北市中正區忠孝東路一段1號,0911111001,user001@example.com
林小芳,1990/07/22,B234567890,新北市板橋區文化路二段50號,0922222002,user002@example.com
...
