# -*- coding: utf-8 -*-
"""
以 E5 向量 + Chroma 做九類公文主旨分類（強化版）
改動重點：
1) 決策從 softmax(prob) 改為「max_sim + margin」雙門檻（可 per-class）
2) 支援規則微加權（RULES）與類別先驗（CLASS_PRIOR）
3) CLI 參數新增：--decision / --thr-sim / --thr-margin / --per-class-thr
4) 重用嵌入器與資料庫連線（大量批次更快）

建庫一致性很重要：
- passage 向量：encode("passage: {text}", normalize_embeddings=True)
- query  向量：encode("query: {text}",   normalize_embeddings=True)
"""
import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

# ========= 參數（依實際建庫一致） =========
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "gov_letters"
EMBED_MODEL = "intfloat/multilingual-e5-large"

# evaluate.py 先前得到的舊組合（仍可對照）
TOPK = 5
TEMPERATURE = 0.7
THRESHOLD = 0.3
RETURN_NEIGHBORS = 5

# ====== 規則微加權（可按需擴增；避免太大，防止壓過語義距離） ======
RULES = {
    "保單查詢＋註記": ["註記", "備註", "加註", "更正註記", "補註記"]
}
RULE_BONUS = 0.05  # 0.02~0.10 之間微調

# ====== 類別先驗（可選）；若不需要可保持空 dict ======
# 想法：若某類樣本特別少，可加一點倍率，例如 1.05~1.15
CLASS_PRIOR: Dict[str, float] = {
    # "保單查詢": 1.0,
    # "保單查詢＋註記": 1.05,
}


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


def load_per_class_thresholds(path: str) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    # 期望格式：{"保單查詢": {"sim": 0.35, "margin": 0.10}, ...}
    return d


@dataclass
class PredResult:
    label: str
    prob: float                     # 依 decision：sim_margin → max_sim；prob → 機率；mixed → 綜合
    neighbors: List[Dict]
    probs: Dict[str, float]         # softmax(votes)（僅供觀察）
    extras: Dict[str, float]        # {"max_sim":..., "margin":..., "p1":...}


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
            f"請確認建庫程式的 PERSIST_DIR/COLLECTION_NAME 是否一致。"
        ) from e
    return col


# ========= kNN + 投票（max_sim + margin 雙門檻） =========
def knn_vote(
    query_vec: np.ndarray,
    collection,
    topk: int = TOPK,
    temperature: float = TEMPERATURE,   # 用於 softmax 顯示
    threshold: float = THRESHOLD,       # 只在 decision="prob"/"mixed" 有用
    return_neighbors: int = RETURN_NEIGHBORS,
    decision: str = "sim_margin",       # "sim_margin" / "prob" / "mixed"
    thr_sim: float = 0.35,              # 全域 max_sim 門檻
    thr_margin: float = 0.08,           # 全域 margin 門檻
    per_class_thr: dict = None,         # {"label": {"sim":..., "margin":...}}
    raw_text: str = "",                 # 規則微加權用
) -> PredResult:
    res = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=topk,
        include=["metadatas", "documents", "distances", "embeddings"],
    )

    metadatas = res.get("metadatas", [[]])[0]
    docs = res.get("documents", [[]])[0]
    distances = res.get("distances", [[]])[0]
    embs = res.get("embeddings", [[]])[0]

    # 1) 取相似度 sims
    if distances:
        d = np.array(distances, dtype=float)
        if np.all((0 <= d) & (d <= 2)):
            sims = 1.0 - d
        else:
            sims = -d
    elif embs:
        neigh = np.array(embs, dtype=np.float32)
        sims = neigh @ query_vec
    else:
        raise RuntimeError("Chroma 回傳缺少 distances/embeddings，無法計分。")

    # 2) 聚合票數 + 每類別的最大相似度
    votes = defaultdict(float)
    label_max_sim = defaultdict(lambda: -1e9)
    neighbors = []
    for i, md in enumerate(metadatas):
        lb = md.get("label", "Other")
        s = float(sims[i])
        votes[lb] += s
        if s > label_max_sim[lb]:
            label_max_sim[lb] = s
        neighbors.append({
            "rank": i + 1,
            "label": lb,
            "score": s,
            "text": docs[i] if i < len(docs) else "",
        })

    # 2.1 類別先驗
    if CLASS_PRIOR:
        for lb in list(votes.keys()):
            votes[lb] *= float(CLASS_PRIOR.get(lb, 1.0))

    # 2.2 規則微加權
    if raw_text and RULES:
        for lb, terms in RULES.items():
            if any(t in raw_text for t in terms):
                votes[lb] += RULE_BONUS

    if not votes:
        probs = {"Other": 1.0}
        return PredResult("Other", 1.0, neighbors[:return_neighbors], probs, {"max_sim": 0.0, "margin": 0.0, "p1": 1.0})

    # 3) top1、top2
    sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
    (lab1, score1) = sorted_votes[0]
    (lab2, score2) = (sorted_votes[1] if len(sorted_votes) > 1 else ("<None>", 0.0))
    margin = float(score1 - score2)
    max_sim_top1 = float(label_max_sim[lab1])

    # 4) softmax 機率（顯示/備用）
    probs = softmax_dict(votes, T=temperature)
    p1 = float(probs.get(lab1, 0.0))

    # 5) 決策
    if decision == "prob":
        label = lab1 if p1 >= threshold else "Other"
        conf = p1
    elif decision == "mixed":
        thr = per_class_thr.get(lab1, {}) if per_class_thr else {}
        ts = float(thr.get("sim", thr_sim))
        tm = float(thr.get("margin", thr_margin))
        ok_sim_margin = (max_sim_top1 >= ts) and (margin >= tm)
        ok_prob = (p1 >= threshold)
        label = lab1 if (ok_sim_margin and ok_prob) else "Other"
        conf = max(p1, max_sim_top1)
    else:  # "sim_margin"
        thr = per_class_thr.get(lab1, {}) if per_class_thr else {}
        ts = float(thr.get("sim", thr_sim))
        tm = float(thr.get("margin", thr_margin))
        label = lab1 if (max_sim_top1 >= ts and margin >= tm) else "Other"
        conf = max_sim_top1

    return PredResult(
        label=label,
        prob=float(conf),
        neighbors=neighbors[:return_neighbors],
        probs=probs,
        extras={"max_sim": max_sim_top1, "margin": margin, "p1": p1},
    )


def classify_subject(
    subject: str,
    embedder: E5Embedder,
    collection,
    topk: int = TOPK,
    temperature: float = TEMPERATURE,
    threshold: float = THRESHOLD,
    return_neighbors: int = RETURN_NEIGHBORS,
    decision: str = "sim_margin",
    thr_sim: float = 0.35,
    thr_margin: float = 0.08,
    per_class_thr: dict = None,
) -> PredResult:
    q = embedder.encode_query(subject)
    return knn_vote(
        q, collection,
        topk=topk, temperature=temperature, threshold=threshold,
        return_neighbors=return_neighbors,
        decision=decision, thr_sim=thr_sim, thr_margin=thr_margin,
        per_class_thr=per_class_thr, raw_text=subject
    )


# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(description="Classify official document subject into 9 labels using E5 + Chroma (強化版).")
    ap.add_argument("--text", type=str, help="要分類的主旨內容（單句）")
    ap.add_argument("--file", type=str, help="逐行讀取檔案中的每一行作為一筆主旨")
    ap.add_argument("--interactive", action="store_true", help="互動模式，逐筆輸入主旨")

    ap.add_argument("--model", type=str, default=EMBED_MODEL, help="SentenceTransformer 模型名稱")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"], help="推論裝置")

    ap.add_argument("--topk", type=int, default=TOPK, help="kNN 取前幾名")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE, help="softmax 溫度（僅顯示用）")
    ap.add_argument("--threshold", type=float, default=THRESHOLD, help="prob/mixed 模式的機率門檻")
    ap.add_argument("--return_neighbors", type=int, default=RETURN_NEIGHBORS, help="回傳前幾個近鄰以供人工檢視")

    ap.add_argument("--decision", type=str, default="sim_margin",
                    choices=["sim_margin", "prob", "mixed"],
                    help="決策方式：sim_margin（建議）、prob（舊式）、mixed（雙重保守）")
    ap.add_argument("--thr-sim", type=float, default=0.35, help="全域 max_sim 門檻，建議 0.30~0.45 掃描")
    ap.add_argument("--thr-margin", type=float, default=0.08, help="全域 margin 門檻（top1 - top2），建議 0.05~0.15 掃描")
    ap.add_argument("--per-class-thr", type=str, default=None,
                    help="每類別門檻 JSON 路徑，格式：{\"保單查詢\":{\"sim\":0.35,\"margin\":0.10}, ...}")

    args = ap.parse_args()

    # 建立嵌入器與 DB（重用以提速）
    embedder = E5Embedder(model_name=args.model, device=args.device)
    collection = get_collection()
    per_class_thr = load_per_class_thresholds(args.per_class_thr)

    def _print_result(text: str, pred: PredResult):
        print("\n====================")
        print(f"🔎 Subject: {text}")
        print(f"👉 Predicted Label: {pred.label}  (conf={pred.prob:.4f})")
        print(f"   [max_sim={pred.extras.get('max_sim', 0):.4f}  margin={pred.extras.get('margin', 0):.4f}  p1={pred.extras.get('p1', 0):.4f}]")
        top_probs = sorted(pred.probs.items(), key=lambda x: -x[1])[:3]
        print("📈 Top-3 probabilities:")
        for k, v in top_probs:
            print(f"  - {k}: {v:.4f}")
        print("—— Neighbors ——")
        for n in pred.neighbors:
            snippet = (n["text"][:100] + "…") if len(n["text"]) > 100 else n["text"]
            print(f"#{n['rank']:>2} [{n['label']}] sim={n['score']:.4f} | {snippet}")

    if args.text:
        pred = classify_subject(
            args.text, embedder, collection,
            topk=args.topk, temperature=args.temperature, threshold=args.threshold,
            return_neighbors=args.return_neighbors, decision=args.decision,
            thr_sim=args.thr_sim, thr_margin=args.thr_margin, per_class_thr=per_class_thr
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
                    line, embedder, collection,
                    topk=args.topk, temperature=args.temperature, threshold=args.threshold,
                    return_neighbors=args.return_neighbors, decision=args.decision,
                    thr_sim=args.thr_sim, thr_margin=args.thr_margin, per_class_thr=per_class_thr
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
                text, embedder, collection,
                topk=args.topk, temperature=args.temperature, threshold=args.threshold,
                return_neighbors=args.return_neighbors, decision=args.decision,
                thr_sim=args.thr_sim, thr_margin=args.thr_margin, per_class_thr=per_class_thr
            )
            _print_result(text, pred)
        return

    # 預設：提示用法
    ap.print_help()


if __name__ == "__main__":
    main()
