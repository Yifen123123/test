# batch_json_to_txt.py
# 批量把 outputs/json/*.json 轉成 outputs/txt/*.txt
# 用法：
#   python batch_json_to_txt.py
# 或指定資料夾：
#   python batch_json_to_txt.py --json_dir outputs/json --txt_dir outputs/txt

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fmt(v: Any) -> str:
    """把 None/空值轉成更好讀的字串。"""
    if v is None:
        return "未提及"
    if isinstance(v, str) and v.strip() == "":
        return "未提及"
    return str(v)


def json_obj_to_txt_lines(data: Dict[str, Any]) -> List[str]:
    lines: List[str] = []

    # ===== 摘要 =====
    lines.append("【摘要】")
    summary = data.get("summary") or []
    if not summary:
        lines.append("無")
    else:
        for item in summary:
            lines.append(f"- {_fmt(item)}")
    lines.append("")

    # ===== 待辦事項 =====
    lines.append("【待辦事項】")
    todo = data.get("todo") or {}
    status = todo.get("status", "unknown")
    lines.append(f"狀態：{_fmt(status)}")

    items = todo.get("items") or []
    if not items:
        lines.append("無")
    else:
        for i, it in enumerate(items, 1):
            desc = _fmt(it.get("description"))
            lines.append(f"{i}. {desc}")

            # flags（用更可讀的方式呈現）
            nf = it.get("need_follow_up")
            nc = it.get("need_callback")
            nt = it.get("need_internal_transfer")

            def b(v: Any) -> str:
                if v is True:
                    return "是"
                if v is False:
                    return "否"
                return "未提及"

            lines.append(f"   - 需追蹤：{b(nf)}；需回電：{b(nc)}；需內轉：{b(nt)}")

            note = it.get("note")
            if note not in (None, ""):
                lines.append(f"   - 備註：{_fmt(note)}")
    lines.append("")

    # ===== 個人資訊 =====
    lines.append("【個人資訊】")
    p = data.get("personal_info") or {}
    lines.append(f"是否為保戶本人：{_fmt(p.get('is_policy_holder'))}")
    lines.append(f"姓名：{_fmt(p.get('name'))}")
    lines.append(f"電話：{_fmt(p.get('phone'))}")

    addr = p.get("address") or {}
    lines.append(f"舊地址：{_fmt(addr.get('old'))}")
    lines.append(f"新地址：{_fmt(addr.get('new'))}")

    lines.append(f"身分證字號：{_fmt(p.get('id_number'))}")
    lines.append(f"出生年月日：{_fmt(p.get('birth_date'))}")
    lines.append(f"保單編號：{_fmt(p.get('policy_number'))}")
    lines.append("")

    # ===== 申訴與不滿 =====
    lines.append("【申訴與不滿】")
    c = data.get("complaint") or {}
    has_c = c.get("has_complaint")
    if has_c is True:
        lines.append("是否有申訴：是")
    elif has_c is False:
        lines.append("是否有申訴：否")
    else:
        lines.append("是否有申訴：未提及")

    desc = c.get("description")
    if desc not in (None, ""):
        lines.append(f"內容：{_fmt(desc)}")

    auth = c.get("mentioned_authority") or []
    if auth:
        lines.append(f"提及單位：{'、'.join(map(str, auth))}")
    else:
        lines.append("提及單位：無")
    lines.append("")

    # ===== 內容描述 =====
    lines.append("【內容描述】")
    narrative = data.get("narrative")
    lines.append(_fmt(narrative))

    return lines


def json_file_to_txt(json_path: Path, txt_path: Path) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{json_path} 內容不是 JSON object（dict）")

    lines = json_obj_to_txt_lines(data)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch convert JSON files to TXT.")
    parser.add_argument("--json_dir", default="outputs/json", help="JSON input directory")
    parser.add_argument("--txt_dir", default="outputs/txt", help="TXT output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing txt files")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    txt_dir = Path(args.txt_dir)

    if not json_dir.exists():
        raise SystemExit(f"找不到 JSON 資料夾：{json_dir}")

    txt_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted([p for p in json_dir.glob("*.json") if p.is_file()])
    if not json_files:
        print(f"{json_dir} 內沒有 .json 檔案")
        return

    ok = 0
    fail = 0

    for jf in json_files:
        tf = txt_dir / f"{jf.stem}.txt"

        if tf.exists() and not args.overwrite:
            # 已存在就跳過
            continue

        try:
            json_file_to_txt(jf, tf)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[FAIL] {jf.name}: {e}")

    print(f"完成：ok={ok}, fail={fail}, output_dir={txt_dir}")


if __name__ == "__main__":
    main()
