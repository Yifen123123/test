import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import uuid

# 1) 載入資料（你可先把 examples.jsonl 轉成 subject,label 的 DataFrame）
df = pd.read_csv("data/labeled.csv")  # columns: subject,label

# 2) 嵌入器（E5：記得 passage:、normalize）
model = SentenceTransformer("intfloat/multilingual-e5-large")
def encode_subjects(texts):
    texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

embs = encode_subjects(df["subject"].tolist())

# 3) 建 Chroma collection
client = chromadb.Client(Settings(anonymized_telemetry=False, persist_directory="chroma_db"))
col = client.get_or_create_collection(name="gov_letters", metadata={"hnsw:space": "ip"}) 
# ip = inner product，搭配 L2-normalized 就是 cosine

# 4) 寫入（每筆一個 id，metadata 帶 label）
ids = [str(uuid.uuid4()) for _ in range(len(df))]
col.add(
    ids=ids,
    embeddings=embs.tolist(),            # list of list(float)
    documents=df["subject"].tolist(),    # 原文可選
    metadatas=[{"label": str(lb)} for lb in df["label"]]
)

# 5) 持久化
client.persist()
print("✅ Chroma index built at ./chroma_db")
