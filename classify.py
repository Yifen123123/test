# classify.py
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

DB_PATH = "chroma_db"
COLL    = "gov_letters"

client = chromadb.PersistentClient(path=DB_PATH)
col = client.get_collection(COLL)

enc = SentenceTransformer("intfloat/multilingual-e5-large")
def embed_one(x):
    return enc.encode([f"passage: {x}"], normalize_embeddings=True)[0].tolist()

def classify_subject(text, k=7, tau=0.55, T=0.10):
    q = embed_one(text)
    res = col.query(query_embeddings=[q], n_results=k, include=["metadatas","documents","distances"])
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    docs  = res["documents"][0]

    sims = np.clip(1 - np.array(dists, dtype=np.float32), 0.0, 1.0)
    logits = sims / T
    w = np.exp(logits - logits.max()); w = w / (w.sum() + 1e-9)

    from collections import defaultdict
    votes = defaultdict(float)
    for i, m in enumerate(metas):
        votes[m["label"]] += float(w[i])

    label, score = max(votes.items(), key=lambda kv: kv[1])
    if score < tau:
        label = "Other"
    return {"label": label, "confidence": float(score),
            "neighbors": list(zip(docs, [m["label"] for m in metas], sims.tolist()))}

print(classify_subject("公告修正補助辦法"))
