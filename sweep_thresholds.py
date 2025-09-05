# -*- coding: utf-8 -*-
"""
掃描 E5+Chroma 分類器的決策門檻：
1) 全域門檻：thr_sim × thr_margin → 找最佳 Macro-F1
2) (可選) 每類別微調：在全域最佳基礎上，逐類別掃 sim 或 margin 的最佳值
輸出：
- best_global.txt / best_per_class.txt
- pred_global.csv / pred_per_class.csv
- per_class_thr.json（若有微調）
- features.parquet（每筆的 top1, max_sim, margin, p1，供重複實驗重用）

需求：pip install pandas numpy scikit-learn pyarrow sentence-transformers chromadb
"""
import argparse, json, re
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, accuracy_score

import chromadb
from sentence_transformers import SentenceTransformer

# ========= 與 classify.py 同步的常數/規則 =========
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "gov_letters"
EMBED_MODEL = "intfloat/multilingual-e5-large"

# 與 classify.py 保持一致（微加權與先驗）
RULES = {
    "保單查詢＋註記": ["註記", "備註", "加註", "更正註記", "補註記"]
}
RULE_BONUS = 0.05
CLASS_PRIOR: Dict[str, float] = {
    # "保單查詢": 1.0,
    # "保單查詢＋註記": 1.05,
}

# ========= 基礎工具 =========
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
    p = np.exp(vals); p /= (p.sum() + 1e-12)
    return {labels[i]: float(p[i]) for i in range(len(labels))}

@dataclass
class CoreFeat:
    text: str
    y_true: str
    top1: str       # 票數最高的 label（未套門檻）
    max_sim: float  # 該 top1 label 的最大相似度
    margin: float   # 票數差：score(top1) - score(top2)
    p1: float       # softmax(votes)[top1]

class E5Embedder:
    def __init__(self, model_name=EMBED_MODEL, device="auto"):
        dev = pick_device(device)
        print(f"🖥️ 使用裝置：{dev}")
        self.model = SentenceTransformer(model_name, device=dev)
    def encode_query(self, text: str) -> np.ndarray:
        emb = self.model.encode([f"query: {text}"], normalize_embeddings=True,
                                batch_size=32, show_progress_bar=False)
        return np.asarray(emb[0], dtype=np.float32)

def get_collection():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        names = [c.name for c in client.list_collections()]
        raise RuntimeError(f"找不到 collection='{COLLECTION_NAME}'，目前有：{names}") from e
    return col

# ========= 單筆推論 → 取核心特徵 =========
def infer_core_feat(text: str, y_true: str, embedder: E5Embedder, collection, topk=5, temperature=0.7) -> CoreFeat:
    q = embedder.encode_query(text)
    res = collection.query(
        query_embeddings=[q.tolist()],
        n_results=topk,
        include=["metadatas", "documents", "distances", "embeddings"]
    )
    md = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    embs = res.get("embeddings", [[]])[0]

    # sims
    if dists:
        d = np.array(dists, dtype=float)
        if np.all((0 <= d) & (d <= 2)):
            sims = 1.0 - d
        else:
            sims = -d
    elif embs:
        neigh = np.array(embs, dtype=np.float32)
        sims = neigh @ q
    else:
        raise RuntimeError("Chroma 回傳缺少 distances/embeddings。")

    from collections import defaultdict
    votes = defaultdict(float)
    label_max_sim = defaultdict(lambda: -1e9)
    for i, m in enumerate(md):
        lb = m.get("label", "Other")
        s = float(sims[i])
        votes[lb] += s
        if s > label_max_sim[lb]:
            label_max_sim[lb] = s

    # 類別先驗
    for lb in list(votes.keys()):
        votes[lb] *= float(CLASS_PRIOR.get(lb, 1.0))
    # 規則微加權
    if text and RULES:
        for lb, terms in RULES.items():
            if any(t in text for t in terms):
                votes[lb] += RULE_BONUS

    if not votes:
        return CoreFeat(text, y_true, "Other", 0.0, 0.0, 1.0)

    sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
    lab1, sc1 = sorted_votes[0]
    lab2, sc2 = (sorted_votes[1] if len(sorted_votes) > 1 else ("<None>", 0.0))
    margin = float(sc1 - sc2)
    max_sim = float(label_max_sim[lab1])
    p = softmax_dict(votes, T=temperature)
    p1 = float(p.get(lab1, 0.0))
    return CoreFeat(text, y_true, lab1, max_sim, margin, p1)

# ========= 用門檻做決策 =========
def decide_label(cf: CoreFeat, decision: str, thr_sim: float, thr_margin: float, prob_thr: float) -> str:
    # sim_margin：max_sim>=thr_sim 且 margin>=thr_margin
    if decision == "sim_margin":
        return cf.top1 if (cf.max_sim >= thr_sim and cf.margin >= thr_margin) else "Other"
    elif decision == "prob":
        return cf.top1 if (cf.p1 >= prob_thr) else "Other"
    else:  # mixed
        ok1 = (cf.max_sim >= thr_sim and cf.margin >= thr_margin)
        ok2 = (cf.p1 >= prob_thr)
        return cf.top1 if (ok1 and ok2) else "Other"

def eval_preds(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, str]:
    labels_set = sorted(list(set(y_true)))  # 僅以真實出現的類別計 macro-F1
    f1 = f1_score(y_true, y_pred, average="macro", labels=labels_set, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    return f1, acc, report

# ========= 解析 best_global.txt =========
def read_best_global(path: Path):
    txt = path.read_text(encoding="utf-8")
    m1 = re.search(r"thr_sim\s*=\s*([0-9.]+)", txt)
    m2 = re.search(r"thr_margin\s*=\s*([0-9.]+)", txt)
    if not (m1 and m2):
        raise ValueError("best_global.txt 格式無法解析 thr_sim/thr_margin")
    return float(m1.group(1)), float(m2.group(1))

# ========= 主程式 =========
def main():
    ap = argparse.ArgumentParser(description="掃描 E5+Chroma 的決策門檻（全域與每類別微調）")
    ap.add_argument("--csv", required=True, help="驗證集 CSV（text,label）")
    ap.add_argument("--decision", type=str, default="sim_margin", choices=["sim_margin","prob","mixed"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--prob-thr", type=float, default=0.3, help="decision=prob/mixed 時的機率門檻")

    # 全域門檻掃描範圍
    ap.add_argument("--thr-sim-min", type=float, default=0.30)
    ap.add_argument("--thr-sim-max", type=float, default=0.45)
    ap.add_argument("--thr-sim-steps", type=int, default=16)
    ap.add_argument("--thr-margin-min", type=float, default=0.05)
    ap.add_argument("--thr-margin-max", type=float, default=0.15)
    ap.add_argument("--thr-margin-steps", type=int, default=11)

    # 重用/輸入
    ap.add_argument("--load-features", type=str, default=None, help="如果已經算過 features.parquet，可用這個直接載入省時間")
    ap.add_argument("--use-global-from", type=str, default=None, help="讀 best_global.txt（含 thr_sim/thr_margin），跳過全域掃描")

    # 每類別微調
    ap.add_argument("--tune-per-class", type=str, default=None, choices=[None,"sim","margin"], help="選擇微調哪個門檻（sim 或 margin）")
    ap.add_argument("--per-class-steps", type=int, default=8, help="單類別掃描步數（在全域 thr 附近做微調）")
    ap.add_argument("--per-class-span", type=float, default=0.06, help="單類別掃描範圍寬度（±span/2）")

    # 其他
    ap.add_argument("--model", type=str, default=EMBED_MODEL)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps","cuda"])
    ap.add_argument("--out-dir", type=str, default="out_sweep")

    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 讀資料
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    if not {"text","label"}.issubset({c.strip().lower() for c in df.columns}):
        raise SystemExit("CSV 必須包含 text,label 欄位")
    df = df.rename(columns={c:c.strip().lower() for c in df.columns})
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # 載入或計算 features（每筆的 top1,max_sim,margin,p1）
    if args.load_features and Path(args.load_features).exists():
        feats = pd.read_parquet(args.load_features)
        print(f"✅ 已載入 features：{len(feats)} 筆，來自 {args.load_features}")
    else:
        embedder = E5Embedder(model_name=args.model, device=args.device)
        col = get_collection()
        feats_list: List[CoreFeat] = []
        for i, row in df.iterrows():
            cf = infer_core_feat(row["text"], row["label"], embedder, col,
                                 topk=args.topk, temperature=args.temperature)
            feats_list.append(cf)
            if (i+1) % 50 == 0:
                print(f"… 已處理 {i+1}/{len(df)}")
        feats = pd.DataFrame([cf.__dict__ for cf in feats_list])
        feats.to_parquet(out_dir / "features.parquet")
        print(f"💾 已存 features 到 {out_dir/'features.parquet'}")

    y_true = feats["y_true"].tolist()

    # ---- 全域門檻 ----
    if args.use_global_from:
        thr_sim, thr_margin = read_best_global(Path(args.use_global_from))
        print(f"🔁 使用既有全域門檻：thr_sim={thr_sim:.4f} thr_margin={thr_margin:.4f}")
        # 直接評估一次
        y_pred = [decide_label(
                    CoreFeat(t, y, tp, ms, mg, p1),
                    args.decision, thr_sim, thr_margin, args.prob_thr
                  )
                  for t,y,tp,ms,mg,p1 in feats[["text","y_true","top1","max_sim","margin","p1"]].itertuples(index=False)]
        f1, acc, report = eval_preds(y_true, y_pred)
        (out_dir / "pred_global.csv").write_text(
            pd.DataFrame({"text":feats["text"],"true":y_true,"top1":feats["top1"],"pred":y_pred,
                          "max_sim":feats["max_sim"],"margin":feats["margin"],"p1":feats["p1"]}).to_csv(index=False),
            encoding="utf-8"
        )
        (out_dir / "best_global.txt").write_text(
            f"decision={args.decision}\nthr_sim={thr_sim:.6f}\nthr_margin={thr_margin:.6f}\n"
            f"macroF1={f1:.6f}\naccuracy={acc:.6f}\n\n{report}",
            encoding="utf-8"
        )
        print(f"⭐ Macro-F1={f1:.4f}  Acc={acc:.4f}")
    else:
        sim_grid = np.linspace(args.thr_sim_min, args.thr_sim_max, args.thr_sim_steps)
        mar_grid = np.linspace(args.thr_margin_min, args.thr_margin_max, args.thr_margin_steps)
        best = (-1.0, None, None, None, None)  # f1, acc, thr_sim, thr_margin, y_pred
        for ts in sim_grid:
            for tm in mar_grid:
                y_pred = [decide_label(
                            CoreFeat(t, y, tp, ms, mg, p1),
                            args.decision, ts, tm, args.prob_thr
                          )
                          for t,y,tp,ms,mg,p1 in feats[["text","y_true","top1","max_sim","margin","p1"]].itertuples(index=False)]
                f1, acc, _ = eval_preds(y_true, y_pred)
                if f1 > best[0]:
                    best = (f1, acc, ts, tm, y_pred)
        f1, acc, thr_sim, thr_margin, y_pred = best
        print(f"🌟 全域最佳：Macro-F1={f1:.4f} Acc={acc:.4f}  thr_sim={thr_sim:.4f} thr_margin={thr_margin:.4f}")
        report = classification_report(y_true, y_pred, digits=4, zero_division=0)
        pd.DataFrame({"text":feats["text"],"true":y_true,"top1":feats["top1"],"pred":y_pred,
                      "max_sim":feats["max_sim"],"margin":feats["margin"],"p1":feats["p1"]}).to_csv(out_dir/"pred_global.csv", index=False, encoding="utf-8")
        (out_dir / "best_global.txt").write_text(
            f"decision={args.decision}\nthr_sim={thr_sim:.6f}\nthr_margin={thr_margin:.6f}\n"
            f"macroF1={f1:.6f}\naccuracy={acc:.6f}\n\n{report}",
            encoding="utf-8"
        )

    # ---- 每類別微調（可選） ----
    if args.tune_per_class:
        base_thr_sim, base_thr_margin = read_best_global(out_dir / "best_global.txt")
        labels = sorted(set(y_true))
        per_thr = {lb: {"sim": base_thr_sim, "margin": base_thr_margin} for lb in labels}

        for lb in labels:
            mask = (feats["top1"] == lb)
            if not mask.any():
                continue
            vals = feats.loc[mask, "max_sim" if args.tune_per_class=="sim" else "margin"].values
            if len(vals) == 0:
                continue
            lo = np.percentile(vals, 20)
            hi = np.percentile(vals, 80)
            span = args.per_class_span
            center = (lo + hi) / 2.0
            cand = np.linspace(center - span/2, center + span/2, args.per_class_steps)
            cand = np.clip(cand, 0, 1)
            best_f1, best_val = -1.0, None
            for v in cand:
                # 應用 per-class 門檻：只有 top1==lb 的樣本用新值，其餘用 base
                y_pred = []
                for t,y,tp,ms,mg,p1 in feats[["text","y_true","top1","max_sim","margin","p1"]].itertuples(index=False):
                    ts = per_thr[tp]["sim"]
                    tm = per_thr[tp]["margin"]
                    if tp == lb:
                        if args.tune_per_class == "sim":
                            ts = float(v)
                        else:
                            tm = float(v)
                    y_pred.append(decide_label(CoreFeat(t,y,tp,ms,mg,p1), args.decision, ts, tm, args.prob_thr))
                f1, _, _ = eval_preds(y_true, y_pred)
                if f1 > best_f1:
                    best_f1, best_val = f1, float(v)
            if best_val is not None:
                per_thr[lb][args.tune_per_class] = best_val
                print(f"🧩 類別「{lb}」最佳 {args.tune_per_class} = {best_val:.4f}  (Macro-F1={best_f1:.4f})")

        # 最終用 per_thr 評估
        def decide_with_per_thr(cf: CoreFeat) -> str:
            ts = per_thr.get(cf.top1, {}).get("sim", base_thr_sim)
            tm = per_thr.get(cf.top1, {}).get("margin", base_thr_margin)
            return decide_label(cf, args.decision, ts, tm, args.prob_thr)

        y_pred = [decide_with_per_thr(CoreFeat(t,y,tp,ms,mg,p1))
                  for t,y,tp,ms,mg,p1 in feats[["text","y_true","top1","max_sim","margin","p1"]].itertuples(index=False)]
        f1, acc, report = eval_preds(y_true, y_pred)

        # 輸出
        with open(out_dir/"per_class_thr.json", "w", encoding="utf-8") as f:
            json.dump(per_thr, f, ensure_ascii=False, indent=2)
        pd.DataFrame({"text":feats["text"],"true":y_true,"top1":feats["top1"],"pred":y_pred,
                      "max_sim":feats["max_sim"],"margin":feats["margin"],"p1":feats["p1"]}).to_csv(out_dir/"pred_per_class.csv", index=False, encoding="utf-8")
        (out_dir / "best_per_class.txt").write_text(
            f"decision={args.decision}\nmacroF1={f1:.6f}\naccuracy={acc:.6f}\n\n{report}",
            encoding="utf-8"
        )
        print(f"🎯 每類別微調完成：Macro-F1={f1:.4f}  Acc={acc:.4f}\n→ per_class_thr.json 已產生於 {out_dir}")
    else:
        print("（略過每類別微調；若需要請加 --tune-per-class sim 或 --tune-per-class margin）")

if __name__ == "__main__":
    main()
