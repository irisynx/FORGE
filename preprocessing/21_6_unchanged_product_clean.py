import pandas as pd
import os
import time

def remove_identical_product_reactants(input_path, output_path, chunk_size=100000):
    """
    清洗 CSV 文件：
    1. 读取每一行反应。
    2. 将 Mapped_Reactants 按 " . " 拆分为列表。
    3. 检查 Mapped_Products 是否与列表中的任意一个反应物完全一致 (字符串严格匹配)。
    4. 如果一致 -> 剔除；如果不一致 -> 保留。
    """

    stats = {"total_read": 0, "kept": 0, "dropped": 0}

    start_time = time.time()
    print(f"🚀 开始清洗: 移除产物与反应物一致的条目")
    print(f"📄 输入文件: {input_path}")
    print(f"💾 输出文件: {output_path}")
    print(f"✂️ 剔除规则: Product String == Any Reactant String")

    if os.path.exists(output_path):
        os.remove(output_path)

    try:

        reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str)

        for i, chunk in enumerate(reader):

            products = chunk["Mapped_Products"].fillna("").astype(str).tolist()
            reactants = chunk["Mapped_Reactants"].fillna("").astype(str).tolist()

            keep_mask = []

            for p_str, r_str in zip(products, reactants):

                should_keep = True

                if not p_str or not r_str:
                    keep_mask.append(True)
                    continue

                clean_p = p_str.strip()

                r_list = [x.strip() for x in r_str.split(" . ")]

                if clean_p in r_list:
                    should_keep = False

                keep_mask.append(should_keep)

            clean_chunk = chunk[keep_mask]

            rows_read = len(chunk)
            rows_kept = len(clean_chunk)
            stats["total_read"] += rows_read
            stats["kept"] += rows_kept
            stats["dropped"] += rows_read - rows_kept

            write_header = i == 0
            mode = "w" if i == 0 else "a"

            clean_chunk.to_csv(output_path, mode=mode, index=False, header=write_header)

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
        print(f"🗑️ 剔除行数 (产物=反应物): {stats['dropped']}")
        print(f"💾 保存行数 (有效数据): {stats['kept']}")
        print(f"📂 新文件位置: {output_path}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 发生错误: {e}")

if __name__ == "__main__":

    input_csv = "ord_data/ord_processed_samples_refined_cleaned_dataset_sane_features_mapped_filtered_yield.csv"

    output_csv = "ord_data/ord_dataset_no_identical_products_20260313.csv"

    if os.path.exists(input_csv):
        remove_identical_product_reactants(input_csv, output_csv)
    else:
        print(f"找不到文件: {input_csv}")
