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

for text, ents in zip(texts, results):
    display(HTML(render_ner_html(text, ents)))
