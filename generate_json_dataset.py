# 置頂記得: import csv
import csv

# 新增：單筆推論但重用 embedder/collection
def classify_with_runtime(
    text: str,
    emb: E5Embedder,
    col,
    topk: int,
    temperature: float,
    threshold: float,
    return_neighbors: int,
) -> PredResult:
    q = emb.encode_query(text)
    return knn_vote(
        q, col,
        topk=topk,
        temperature=temperature,
        threshold=threshold,
        return_neighbors=return_neighbors,
    )

# ===== 替換原本的 if args.file: 區塊 =====
if args.file:
    # 1) 只載入一次模型與資料庫
    emb = E5Embedder(model_name=args.model, device=args.device)  # 這裡只會印一次「使用裝置：...」
    col = get_collection()

    # 2) 準備輸出 CSV
    with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["text", "label"])  # 表頭

        # 3) 逐行讀取並分類（重用 emb/col）
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    text = line.strip()
                    if not text:
                        continue
                    pred = classify_with_runtime(
                        text, emb, col,
                        topk=args.topk,
                        temperature=args.temperature,
                        threshold=args.threshold,
                        return_neighbors=0  # 不取鄰居，加速
                    )
                    writer.writerow([text, pred.label])

                    # 可選：每 N 筆 flush 一次，避免中途崩潰資料沒落盤
                    if i % 50 == 0:
                        csvf.flush()
        except KeyboardInterrupt:
            print("\n⏹️ 手動中斷，已保留到目前寫入的 CSV。")
        except Exception as e:
            print(f"\n❌ 發生錯誤：{e}\n已保留到目前寫入的 CSV。")
            raise

    print(f"✅ 已完成批次分類，CSV 輸出到：{args.out_csv}")
    return
