# build_index.py
import json
import uuid
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ============ 參數 ============
JSONL_PATH = "data/examples.jsonl"         # 每行一個 {"label": "...", "text": "..."}
PERSIST_DIR = "chroma_db"                   # Chroma 持久化目錄
COLLECTION_NAME = "gov_letters"             # 你的 collection 名稱
EMBED_MODEL = "intfloat/multilingual-e5-large"

# ============ 讀取 JSONL ============
def load_jsonl(jsonl_path: str):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # 需要有 "label" 與 "text"
            if "label" in obj and "text" in obj:
                items.append(obj)
    if not items:
        raise ValueError("JSONL 讀不到任何有效資料（需每行一個含 label/text 的 JSON 物件）")
    return items

# ============ 建立向量 ============
def get_embedder():
    # E5：資料庫側用 passage: 前綴；normalize=True
    model = SentenceTransformer(EMBED_MODEL)
    def encode_passages(texts):
        texts = [f"passage: {t}" for t in texts]
        return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    return encode_passages

def main():
    data = load_jsonl(JSONL_PATH)
    texts = [d["text"] for d in data]
    labels = [str(d["label"]) for d in data]

    encode_passages = get_embedder()
    embs = encode_passages(texts)  # ndarray (N, D)

    # ============ 建立/讀取 Chroma collection ============
    # 用 ip（inner product）+ 向量 L2-normalized => 等價 cosine
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "ip"}  # 也可用 "cosine"；ip + normalize = cosine
    )

    # ============ 寫入 ============
    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    col.add(
        ids=ids,
        embeddings=embs.tolist(),
        documents=texts,  # 原文存一下，debug/回傳用
        metadatas=[{"label": lb} for lb in labels]
    )

    # ============ 持久化 ============
    client.persist()
    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    print(f"✅ Index 構建完成，已持久化於 ./{PERSIST_DIR}，collection = {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
