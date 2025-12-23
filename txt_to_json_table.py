import re
import json


def parse_investigators_txt(txt: str):
    """
    將你提供的 txt 轉成：
    [
      { "調查人資料": [ {姓名, 身分證字號, 財產基準日} x5 ] },
      ...
    ]
    """
    cases = []

    # 1️⃣ 先切案件
    raw_cases = re.split(r"＝{3,}", txt)

    for raw_case in raw_cases:
        raw_case = raw_case.strip()
        if not raw_case:
            continue

        investigators = []

        # 2️⃣ 切調查人（ㄧ、二、三、四、五 都吃）
        people_blocks = re.split(r"調查人[ㄧ一二三四五]", raw_case)

        for block in people_blocks:
            block = block.strip()
            if not block:
                continue

            def extract(label: str) -> str:
                """
                從 block 中擷取：
                姓名：
                身分證字號：
                財產基準日：
                """
                m = re.search(rf"{label}：([^\n]*)", block)
                return m.group(1).strip() if m else ""

            person = {
                "姓名": extract("姓名"),
                "身分證字號": extract("身分證字號"),
                "財產基準日": extract("財產基準日"),
            }

            investigators.append(person)

        # 安全檢查：你預期每個案件 5 人
        if investigators:
            cases.append({
                "調查人資料": investigators
            })

    return cases


def main():
    # 讀你的 txt
    with open("investigators.txt", "r", encoding="utf-8") as f:
        txt = f.read()

    data = parse_investigators_txt(txt)

    # 存成 JSON（你後面就能直接用）
    with open("investigators.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 轉換完成，共 {len(data)} 筆案件")
    print("已輸出 investigators.json")


if __name__ == "__main__":
    main()
