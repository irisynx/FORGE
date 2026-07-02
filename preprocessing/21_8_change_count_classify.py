import pandas as pd
import os
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

def get_heavy_atom_topology(smiles_str):
    """
    解析 SMILES，提取每个映射原子的【重原子邻居集合】。
    完全忽略氢原子 (AtomicNum=1)。
    """
    topology = {}
    if not isinstance(smiles_str, str) or not smiles_str.strip():
        return None

    fragments = smiles_str.replace(" . ", ".").split(".")

    all_frags_valid = False

    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue

        mol = Chem.MolFromSmiles(frag, sanitize=False)
        if mol is None:
            continue

        try:
            mol.UpdatePropertyCache(strict=False)
            all_frags_valid = True
        except:
            continue

        for atom in mol.GetAtoms():

            if atom.GetAtomicNum() == 1:
                continue

            map_id = atom.GetAtomMapNum()
            if map_id == 0:
                continue

            heavy_neighbors = set()
            for nbr in atom.GetNeighbors():

                if nbr.GetAtomicNum() == 1:
                    continue

                nbr_id = nbr.GetAtomMapNum()
                if nbr_id > 0:
                    heavy_neighbors.add(nbr_id)

            topology[map_id] = heavy_neighbors

    if not all_frags_valid:
        return None

    return topology

def filter_reactions_by_complexity(
    input_path, output_clean, output_removed, chunk_size=100000
):

    THRESHOLD = 5

    stats = {"total": 0, "kept": 0, "removed_high_change": 0, "removed_parse_error": 0}

    print(f"🚀 开始严格筛选 (阈值 < {THRESHOLD}, 忽略氢原子)")
    print(f"📄 输入: {input_path}")
    print(f"✅ 保留: {output_clean}")
    print(f"🗑️ 剔除: {output_removed}")

    if os.path.exists(output_clean):
        os.remove(output_clean)
    if os.path.exists(output_removed):
        os.remove(output_removed)

    try:
        reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str)

        for i, chunk in enumerate(reader):
            clean_rows = []
            removed_rows = []

            for index, row in chunk.iterrows():
                stats["total"] += 1

                r_str = row.get("Mapped_Reactants", "")
                p_str = row.get("Mapped_Products", "")

                r_topo = get_heavy_atom_topology(r_str)
                p_topo = get_heavy_atom_topology(p_str)

                if r_topo is None or p_topo is None:
                    stats["removed_parse_error"] += 1
                    row["filter_reason"] = "parse_error"
                    removed_rows.append(row)
                    continue

                common_ids = set(r_topo.keys()) & set(p_topo.keys())

                if not common_ids:

                    stats["removed_parse_error"] += 1
                    row["filter_reason"] = "no_common_atoms"
                    removed_rows.append(row)
                    continue

                num_changed = 0
                for mid in common_ids:

                    if r_topo[mid] != p_topo[mid]:
                        num_changed += 1

                if num_changed < THRESHOLD:

                    clean_rows.append(row)
                else:

                    stats["removed_high_change"] += 1
                    row["filter_reason"] = f"high_change_count_{num_changed}"
                    removed_rows.append(row)

            if clean_rows:
                df_clean = pd.DataFrame(clean_rows)
                mode = "a" if os.path.exists(output_clean) else "w"
                header = not os.path.exists(output_clean)
                df_clean.to_csv(output_clean, mode=mode, index=False, header=header)

            if removed_rows:
                df_removed = pd.DataFrame(removed_rows)

                cols = list(df_removed.columns)
                if "filter_reason" not in cols:

                    df_removed["filter_reason"] = "unknown"

                mode = "a" if os.path.exists(output_removed) else "w"
                header = not os.path.exists(output_removed)
                df_removed.to_csv(output_removed, mode=mode, index=False, header=header)

            stats["kept"] += len(clean_rows)

            if (i + 1) % 10 == 0:
                print(
                    f"   ...已处理 {stats['total']} | 保留: {stats['kept']} ({stats['kept']/stats['total']:.1%})"
                )

        print("\n" + "=" * 50)
        print("✅ 筛选完成")
        print(f"📥 总数: {stats['total']}")
        print(
            f"💾 保留 (Clean):   {stats['kept']} ({stats['kept']/stats['total']*100:.2f}%)"
        )
        print(f"🗑️ 剔除 (Removed): {stats['total'] - stats['kept']}")
        print("-" * 30)
        print("🔍 剔除详情:")
        print(f"   🌪️ 剧烈变动 (>= {THRESHOLD}): {stats['removed_high_change']}")
        print(f"   ❌ 解析/映射错误:    {stats['removed_parse_error']}")
        print("=" * 50)

    except Exception as e:
        print(f"❌ 程序出错: {e}")

if __name__ == "__main__":
    input_csv = (
        "/data/chara/project/AI4S/ord_data/23260313_2_ord_dataset_robust_mapped.csv"
    )

    output_clean = "/data/chara/project/AI4S/ord_data/23260313_3_ord_dataset_clean4.csv"
    output_removed = (
        "/data/chara/project/AI4S/ord_data/ord_dataset_removed_complex_up5.csv"
    )

    if os.path.exists(input_csv):
        filter_reactions_by_complexity(input_csv, output_clean, output_removed)
    else:
        print(f"找不到文件: {input_csv}")
