    if args.file:
        # 準備輸出 CSV
        with open(args.out_csv, "w", encoding="utf-8-sig", newline="") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["text", "label"])  # 表頭

            # 逐行讀取並分類
            with open(args.file, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    pred = classify_subject(
                        text,
                        model_name=args.model,
                        device=args.device,
                        topk=args.topk,
                        temperature=args.temperature,
                        threshold=args.threshold,
                        return_neighbors=0  # 不需要鄰居，加速
                    )
                    writer.writerow([text, pred.label])

        print(f"✅ 已完成批次分類，CSV 輸出到：{args.out_csv}")
        return
