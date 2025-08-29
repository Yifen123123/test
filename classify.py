import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# 啟動時載入
client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory="chroma_db"))
col = client.get_collection("gov_letters")
model = SentenceTransformer("intfloat/multilingual-e5-large")

def embed(texts):
    texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

def classify_subject(subject: str, k: int = 7, tau: float = 0.55):
    q = embed([subject])[0].tolist()
    qres = col.query(query_embeddings=[q], n_results=k, include=["metadatas","documents","distances"])
    # Chroma 的 distances 對 IP 會回傳 (1 - cosine) 或者是相似度的轉換；保守起見用官方返回的 "distances"
    # 若你設 "ip"，多半 distances 越小越近；我們轉成相似度 s = 1 - d 做權重
    dists = qres["distances"][0]
    metas = qres["metadatas"][0]
    docs  = qres["documents"][0]

    # 距離轉權重（避免負值/極端）
    sims = np.clip(1 - np.array(dists, dtype=np.float32), 0.0, 1.0)
    # 溫度縮放（可調 T=0.07~0.2），讓權重分佈更尖銳
    T = 0.10
    logits = sims / T
    w = np.exp(logits - logits.max())
    w = w / (w.sum() + 1e-9)

    # 加權投票
    votes = defaultdict(float)
    for i, m in enumerate(metas):
        votes[m["label"]] += float(w[i])

    # 取得最可能類別與信心（最大權重）
    label, score = max(votes.items(), key=lambda x: x[1])
    if score < tau:
        return {"label":"Other", "confidence": float(score), "topk": list(votes.items()), "neighbors": list(zip(docs, [m["label"] for m in metas], sims.tolist()))}
    return {"label":label, "confidence": float(score), "topk": list(votes.items()), "neighbors": list(zip(docs, [m["label"] for m in metas], sims.tolist()))}

# 測試
print(classify_subject("公告修正補助辦法"))
