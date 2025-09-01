from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import json, uuid

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "gov_letters"
JSONL_PATH = "data/examples2.jsonl"
EMBED_MODEL = "intfloat/multilingual-e5-large"

def load_jsonl(p):
    items=[]
    with open(p,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            o=json.loads(line); items.append(o)
    return items

def encode_passages(texts):
    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)

def main():
    data = load_jsonl(JSONL_PATH)
    texts = [d["text"] for d in data]
    labels = [str(d["label"]) for d in data]

    embs = encode_passages(texts)

    client = chromadb.PersistentClient(path=PERSIST_DIR)
    col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "ip"}  # normalize + ip 等同 cosine
    )

    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    col.add(
        ids=ids,
        embeddings=embs.tolist(),
        documents=texts,
        metadatas=[{"label": lb} for lb in labels],
    )

    Path(PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    print("Collections:", [c.name for c in client.list_collections()])
    print(f"✅ Index 構建完成，持久化於 ./{PERSIST_DIR} ，collection = {COLLECTION_NAME}，共 {len(texts)} 筆。")

if __name__ == "__main__":
    main()
