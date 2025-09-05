# predict.py
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from joblib import load
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

EMB_MODEL = "intfloat/multilingual-e5-large"

def embed_texts(texts, batch_size=64):
    model = SentenceTransformer(EMB_MODEL)
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

def load_rules(yaml_path: Path):
    if not yaml_path.exists():
        return {}
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

def nudge_logits(text: str, labels: list[str], logits: np.ndarray, rules: dict) -> np.ndarray:
    """
    rules 形式：
      target_label:
        patterns: ["扣押", "支付轉給命令", "執行處"]
        delta: 0.5
    """
    adj = logits.copy()
    for lab, cfg in rules.items():
        pats = cfg.get("patterns", [])
        delta = float(cfg.get("delta", 0.0))
        if any(p in text for p in pats):
            try:
                idx = labels.index(lab)
                adj[idx] += delta
            except ValueError:
                pass
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, default="data/processed/records.csv")
    ap.add_argument("--text_col", type=str, default="built_text")
    ap.add_argument("--models_dir", type=str, default="models")
    ap.add_argument("--rules_yaml", type=str, default="nudge_rules.yaml")
    ap.add_argument("--out_csv", type=str, default="data/processed/predictions.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    texts = df[args.text_col].astype(str).tolist()

    clf = load(Path(args.models_dir) / "linear_clf.joblib")
    labels = json.loads((Path(args.models_dir)/"labels.json").read_text(encoding="utf-8"))
    X = embed_texts(texts)

    # 原始 logits（scikit 取 predict_proba 前的 decision_function 比較合理；
    # 但 CalibratedClassifierCV 沒有暴露；這裡直接用 predict_proba 的 log 做微調也可）
    probs = clf.predict_proba(X)
    logits = np.log(np.clip(probs, 1e-9, 1.0))

    rules = load_rules(Path(args.rules_yaml))
    nudged = []
    for t, logit in zip(texts, logits):
        adj = nudge_logits(t, labels, logit, rules) if rules else logit
        nudged.append(adj)
    nudged = np.stack(nudged, axis=0)

    preds = nudged.argmax(axis=1)
    pred_labels = [labels[i] for i in preds]
    max_p = nudged.max(axis=1)  # 近似信心分數（注意是logit，僅供排序/相對比較）

    df_out = df.copy()
    df_out["pred"] = pred_labels
    df_out["score_like"] = max_p
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] 已輸出預測：{out_path}")

if __name__ == "__main__":
    main()
