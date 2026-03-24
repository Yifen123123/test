import spacy
import pandas as pd
from spacy import displacy

# 1. 載入模型
nlp_zh = spacy.load("zh_core_web_trf")

# 2. 輸入句子
text = ""
doc = nlp_zh(text)

# 3. 整理成表格
columns = ['詞', '詞類', '詞性標注', '單詞依存關係', '是否為純字母組成', '是否為停用詞']

dim = [
    [token.text, token.pos_, token.tag_, token.dep_, token.is_alpha, token.is_stop]
    for token in doc
]

df = pd.DataFrame(dim, columns=columns)
df['是否為純字母組成'] = df['是否為純字母組成'].map({True: '是', False: '否'})
df['是否為停用詞'] = df['是否為停用詞'].map({True: '是', False: '否'})

# 4. 表格 HTML
table_html = df.to_html(index=False, escape=False)

table_page = f"""
<html>
<head>
    <meta charset="UTF-8">
    <title>spaCy 表格可視化</title>
    <style>
        body {{
            font-family: Arial, "Microsoft JhengHei", sans-serif;
            margin: 30px;
        }}
        h2 {{
            color: #333;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 20px;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 8px 12px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        tr:nth-child(even) {{
            background-color: #fafafa;
        }}
    </style>
</head>
<body>
    <h2>spaCy 中文分析表格</h2>
    <p><b>原句：</b>{text}</p>
    {table_html}
</body>
</html>
"""

with open("spacy_table.html", "w", encoding="utf-8") as f:
    f.write(table_page)

# 5. 依存關係 HTML
dep_html = displacy.render(doc, style="dep", page=True)

with open("spacy_dep.html", "w", encoding="utf-8") as f:
    f.write(dep_html)

print("已產生：spacy_table.html")
print("已產生：spacy_dep.html")
