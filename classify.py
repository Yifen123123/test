# classify.py
# 以 E5 向量 + Chroma 做九類公文主旨分類
# 預設參數來自 evaluate.py 的最佳組合：K=5, T=0.7, THRESHOLD=0.3

import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ========= 參數（依實際建庫一致） =========
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "gov_letters"
EMBED_MODEL = "intfloat/multilingual-e5-large"

# 來自你的 evaluate 最佳組合
TOPK = 5
TEMPERATURE = 0.7
THRESHOLD = 0.3
RETURN_NEIGHBORS = 5


# ========= 工具 =========
def pick_device(name: str = "auto") -> str:
    name = (name or "auto").lower()
    try:
        import torch
    except Exception:
        return "cpu"

    if name != "auto":
        return name

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def softmax_dict(votes: Dict[str, float], T: float = 1.0) -> Dict[str, float]:
    if not votes:
        return {"Other": 1.0}
    labels = list(votes.keys())
    vals = np.array([votes[l] for l in labels], dtype=np.float64)
    vals = vals / max(T, 1e-6)
    vals -= vals.max()
    p = np.exp(vals)
    p /= (p.sum() + 1e-12)
    return {labels[i]: float(p[i]) for i in range(len(labels))}


@dataclass
class PredResult:
    label: str
    prob: float
    neighbors: List[Dict]
    probs: Dict[str, float]


# ========= 模型 & DB =========
class E5Embedder:
    def __init__(self, model_name: str = EMBED_MODEL, device: str = "auto"):
        dev = pick_device(device)
        print(f"🖥️ 使用裝置：{dev}")
        self.model = SentenceTransformer(model_name, device=dev)

    def encode_query(self, text: str) -> np.ndarray:
        inputs = [f"query: {text}"]
        embs = self.model.encode(inputs, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return np.asarray(embs[0], dtype=np.float32)


def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        names = [c.name for c in client.list_collections()]
        raise RuntimeError(
            f"找不到 collection='{COLLECTION_NAME}'，目前可用 collections={names}；"
            f"請確認 build_index.py 的 PERSIST_DIR/COLLECTION_NAME 是否一致。"
        ) from e
    return col


# ========= kNN + 投票（含 T 與門檻） =========
def knn_vote(
    query_vec: np.ndarray,
    collection,
    topk: int = TOPK,
    temperature: float = TEMPERATURE,
    threshold: float = THRESHOLD,
    return_neighbors: int = RETURN_NEIGHBORS,
) -> PredResult:
    res = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=topk,
        include=["metadatas", "documents", "distances", "embeddings"],  # 兼容不同回傳
    )

    metadatas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]
    distances = res.get("distances", [[]])[0]
    embs = res.get("embeddings", [[]])[0]

    # 將 distances 轉為「越大越相似」的 sims
    sims = None
    if distances:
        d = np.array(distances, dtype=float)
        # 常見：cosine distance = 1 - cos ∈ [0,2]；若符合，cos = 1 - d
        if np.all((0 <= d) & (d <= 2)):
            sims = 1.0 - d
        else:
            # 不明格式時，取負距離當分數（距離越小越近）
            sims = -d
    else:
        # 沒有 distances 時，若有回傳近鄰 embedding，自己算內積（已 normalize = cosine）
        if embs:
            neigh = np.array(embs, dtype=np.float32)  # (k, D)
            sims = neigh @ query_vec
        else:
            raise RuntimeError("Chroma 回傳缺少 distances/embeddings，無法計分。請更新 include 參數或 Chroma 版本。")

    votes = defaultdict(float)
    neighbors = []
    for i, md in enumerate(metadatas):
        lb = md.get("label", "Other")
        score = float(sims[i])
        votes[lb] += score
        neighbors.append({
            "rank": i + 1,
            "label": lb,
            "score": score,
            "text": docs[i] if i < len(docs) else "",
        })

    probs = softmax_dict(votes, T=temperature)
    label, prob = max(probs.items(), key=lambda x: x[1])

    if prob < threshold:
        label = "Other"

    return PredResult(
        label=label,
        prob=float(prob),
        neighbors=neighbors[:return_neighbors],
        probs=probs,
    )


def classify_subject(
    subject: str,
    model_name: str = EMBED_MODEL,
    device: str = "auto",
    topk: int = TOPK,
    temperature: float = TEMPERATURE,
    threshold: float = THRESHOLD,
    return_neighbors: int = RETURN_NEIGHBORS,
) -> PredResult:
    emb = E5Embedder(model_name=model_name, device=device)
    q = emb.encode_query(subject)
    col = get_collection()
    return knn_vote(q, col, topk=topk, temperature=temperature, threshold=threshold, return_neighbors=return_neighbors)


# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(description="Classify official document subject into 9 labels using E5 + Chroma.")
    ap.add_argument("--text", type=str, help="要分類的主旨內容（單句）")
    ap.add_argument("--file", type=str, help="逐行讀取檔案中的每一行作為一筆主旨")
    ap.add_argument("--interactive", action="store_true", help="互動模式，逐筆輸入主旨")
    ap.add_argument("--model", type=str, default=EMBED_MODEL, help="SentenceTransformer 模型名稱")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="推論裝置")
    ap.add_argument("--topk", type=int, default=TOPK, help="kNN 取前幾名")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE, help="溫度縮放 T")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="拒答門檻（max_prob < threshold → Other）")
    ap.add_argument("--return_neighbors", type=int, default=RETURN_NEIGHBORS, help="回傳前幾個近鄰以供人工檢視")
    args = ap.parse_args()

    def _print_result(text: str, pred: PredResult):
        print("\n====================")
        print(f"🔎 Subject: {text}")
        print(f"👉 Predicted Label: {pred.label}  (prob={pred.prob:.4f})")

        # 顯示前 3 名機率
        top_probs = sorted(pred.probs.items(), key=lambda x: -x[1])[:3]
        print("📈 Top-3 probabilities:")
        for k, v in top_probs:
            print(f"  - {k}: {v:.4f}")

        # 近鄰檢視
        print("—— Neighbors ——")
        for n in pred.neighbors:
            snippet = (n["text"][:100] + "…") if len(n["text"]) > 100 else n["text"]
            print(f"#{n['rank']:>2} [{n['label']}] sim={n['score']:.4f} | {snippet}")

    if args.text:
        pred = classify_subject(
            args.text, model_name=args.model, device=args.device,
            topk=args.topk, temperature=args.temperature,
            threshold=args.threshold, return_neighbors=args.return_neighbors
        )
        _print_result(args.text, pred)
        return

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pred = classify_subject(
                    line, model_name=args.model, device=args.device,
                    topk=args.topk, temperature=args.temperature,
                    threshold=args.threshold, return_neighbors=args.return_neighbors
                )
                _print_result(line, pred)
        return

    if args.interactive:
        print("互動模式已啟動，輸入空行離開。")
        while True:
            try:
                text = input("\n輸入主旨 > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not text:
                print("Bye.")
                break
            pred = classify_subject(
                text, model_name=args.model, device=args.device,
                topk=args.topk, temperature=args.temperature,
                threshold=args.threshold, return_neighbors=args.return_neighbors
            )
            _print_result(text, pred)
        return

    # 預設：提示用法
    ap.print_help()


if __name__ == "__main__":
    main()
