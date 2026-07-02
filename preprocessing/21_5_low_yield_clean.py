import pandas as pd
import os
import time

def filter_low_yield_rows(input_path, output_path, threshold=10, chunk_size=100000):
    """
    清洗 CSV 文件：
    1. 剔除 Yield < threshold (默认 10) 的行。
    2. 保留 Yield == -1 (原始数据缺失) 的行。
    3. 保留 Yield >= threshold 的行。
    4. 原样保留其他所有列的数据。
    """

    stats = {"total_read": 0, "kept": 0, "dropped": 0}

    start_time = time.time()
    print(f"🚀 开始清洗任务")
    print(f"📄 输入文件: {input_path}")
    print(f"💾 输出文件: {output_path}")
    print(f"✂️ 剔除规则: 剔除 0 <= Yield < {threshold} 的行 (保留 -1)")

    if os.path.exists(output_path):
        os.remove(output_path)

    try:

        reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str)

        for i, chunk in enumerate(reader):

            yield_numeric = pd.to_numeric(chunk["Yield"], errors="coerce")

            condition_keep_neg1 = yield_numeric == -1

            condition_keep_high = yield_numeric >= threshold

            keep_mask = condition_keep_neg1 | condition_keep_high

            filtered_chunk = chunk[keep_mask]

            rows_read = len(chunk)
            rows_kept = len(filtered_chunk)
            stats["total_read"] += rows_read
            stats["kept"] += rows_kept
            stats["dropped"] += rows_read - rows_kept

            write_header = i == 0
            mode = "w" if i == 0 else "a"

            filtered_chunk.to_csv(
                output_path, mode=mode, index=False, header=write_header
            )

            if (i + 1) % 10 == 0:
                print(
                    f"   ...已处理 {stats['total_read']} 行 | 剔除: {stats['dropped']} | 保留: {stats['kept']}"
                )

        end_time = time.time()
        duration = end_time - start_time

        print("\n" + "=" * 50)
        print("✅ 清洗完成！")
        print(f"⏱️ 耗时: {duration:.2f} 秒")
        print(f"📥 读取总行数: {stats['total_read']}")
        print(f"🗑️ 剔除行数 (Yield < {threshold}): {stats['dropped']}")
        print(f"💾 保存行数 (Yield >= {threshold} 或 -1): {stats['kept']}")
        print(f"📂 新文件位置: {output_path}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":

    input_csv = "ord_data/ord_processed_samples_refined_cleaned_dataset_sane_features_mapped.csv"

    output_csv = "ord_data/ord_processed_samples_refined_cleaned_dataset_sane_features_mapped_filtered_yield.csv"

    if os.path.exists(input_csv):
        filter_low_yield_rows(input_csv, output_csv, threshold=10)
    else:
        print(f"找不到文件: {input_csv}")
