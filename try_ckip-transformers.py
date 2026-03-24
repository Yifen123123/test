from IPython.display import display, HTML
from ckip_transformers.nlp import CkipNerChunker

texts = [""]
ner_driver = CkipNerChunker()
results = ner_driver(texts)

label_colors = {
    "PERSON": "#ffd54f",
    "ORG": "#81d4fa",
    "GPE": "#c5e1a5",
    "LOC": "#f8bbd0",
}

def render_ner_html(text, ents):
    html = ""
    last_idx = 0
    for ent in sorted(ents, key=lambda x: x.idx[0]):
        start, end = ent.idx
        html += text[last_idx:start]
        color = label_colors.get(ent.ner, "#e0e0e0")
        html += (
            f"<span style='background:{color}; padding:2px 4px; margin:1px; "
            f"border-radius:4px;'>"
            f"{text[start:end]} <b>({ent.ner})</b></span>"
        )
        last_idx = end
    html += text[last_idx:]
    return html

all_html = """
<html>
<head>
    <meta charset="UTF-8">
    <title>CKIP NER 可視化結果</title>
</head>
<body>
    <h2>CKIP NER 可視化結果</h2>
"""

for text, ents in zip(texts, results):
    all_html += f"<p>{render_ner_html(text, ents)}</p>"

all_html += """
</body>
</html>
"""

with open("ner_result.html", "w", encoding="utf-8") as f:
    f.write(all_html)

print("已輸出 HTML 檔：ner_result.html")
