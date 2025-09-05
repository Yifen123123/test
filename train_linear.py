# train_linear.py
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score
from sentence_transformers import SentenceTransformer

EMB_MODEL = "intfloat/multilingual-e5-large"

def embed_texts(texts, batch_size=64):
    model = SentenceTransformer(EMB_MODEL)
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    )
    return vecs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="data/processed/records.csv")
    ap.add_argument("--text_col", type=str, default="built_text")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--out_dir", type=str, default="models")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    df = df.dropna(subset=[args.text_col, args.label_col])
    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()

    # 轉成 0..K-1
    uniq = sorted(set(labels))
    lab2id = {lab:i for i, lab in enumerate(uniq)}
    y = np.array([lab2id[l] for l in labels], dtype=int)

    X = embed_texts(texts)

    base = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)

    # 簡易自評（train 上；實務請切 train/valid）
    preds = clf.predict(X)
    print(classification_report(y, preds, target_names=uniq, digits=4))
    print("macro-F1:", f1_score(y, preds, average="macro"))

    dump(clf, out_dir / "linear_clf.joblib")
    (out_dir / "labels.json").write_text(json.dumps(uniq, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] 模型已存到 {out_dir}")

if __name__ == "__main__":
    main()
