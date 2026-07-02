import pandas as pd
import os
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

def parse_component_atoms(smiles_str):
    """
    核心解析函数：
    1. 兼容 ' . ' 和 '.' 两种分隔符。
    2. 逐个碎片解析，防止因单个碎片错误导致整体失败。
    3. 汇总所有原子的 MapID 信息。

    返回:
      - atom_dict: { map_id: {'atomic_num': int, 'neighbors': set(neighbor_map_ids)} }
      - parse_success: bool (是否所有非空碎片都解析成功)
    """
    atom_dict = {}
    if not isinstance(smiles_str, str) or not smiles_str.strip():
        return atom_dict, False

    fragments = smiles_str.replace(" . ", ".").split(".")

    all_parsed = True

    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue

        mol = Chem.MolFromSmiles(frag, sanitize=False)

        if mol is None:

            mol = Chem.MolFromSmiles(frag)
            if mol is None:
                all_parsed = False
                continue

        try:
            mol.UpdatePropertyCache(strict=False)
        except:
            pass

        for atom in mol.GetAtoms():
            map_id = atom.GetAtomMapNum()
            if map_id == 0:
                continue

            neighbors = set()
            for nbr in atom.GetNeighbors():
                nbr_id = nbr.GetAtomMapNum()
                if nbr_id > 0:
                    neighbors.add(nbr_id)

            atom_dict[map_id] = {
                "atomic_num": atom.GetAtomicNum(),
                "neighbors": neighbors,
            }

    return atom_dict, all_parsed

def robust_reaction_filter(input_path, output_path, chunk_size=100000):

    MAX_CHANGED_ATOMS = 10

    stats = {
        "total": 0,
        "kept": 0,
        "err_parse": 0,
        "err_magic": 0,
        "err_alchemy": 0,
        "err_scramble": 0,
    }

    print(f"🚀 开始稳健版清洗 (Split & Merge 模式)")
    print(f"📄 输入: {input_path}")
    print(
        f"⚙️ 规则: 允许离去基团 | 禁止无中生有 | 允许 {MAX_CHANGED_ATOMS} 个原子环境改变"
    )

    if os.path.exists(output_path):
        os.remove(output_path)

    debug_errors = []

    try:
        reader = pd.read_csv(input_path, chunksize=chunk_size, dtype=str)

        for i, chunk in enumerate(reader):
            valid_rows = []

            for index, row in chunk.iterrows():
                stats["total"] += 1
                r_str = row.get("Mapped_Reactants", "")
                p_str = row.get("Mapped_Products", "")

                r_atoms, r_ok = parse_component_atoms(r_str)

                p_atoms, p_ok = parse_component_atoms(p_str)

                if not r_atoms or not p_atoms:
                    stats["err_parse"] += 1
                    continue

                keep_row = True
                fail_reason = ""
                connectivity_changes = 0

                for pid, p_info in p_atoms.items():

                    if pid not in r_atoms:
                        keep_row = False
                        stats["err_magic"] += 1
                        fail_reason = (
                            f"Magic Matter: ID {pid} in Product but not Reactant"
                        )
                        break

                    r_info = r_atoms[pid]

                    if r_info["atomic_num"] != p_info["atomic_num"]:
                        keep_row = False
                        stats["err_alchemy"] += 1
                        fail_reason = f"Alchemy: ID {pid} changed from {r_info['atomic_num']} to {p_info['atomic_num']}"
                        break

                    if r_info["neighbors"] != p_info["neighbors"]:
                        connectivity_changes += 1

                if not keep_row:
                    if len(debug_errors) < 5:
                        debug_errors.append((stats["total"], fail_reason))
                    continue

                if connectivity_changes > MAX_CHANGED_ATOMS:
                    stats["err_scramble"] += 1
                    fail_reason = f"Scrambling: {connectivity_changes} atoms changed connectivity (> {MAX_CHANGED_ATOMS})"
                    if len(debug_errors) < 5:
                        debug_errors.append((stats["total"], fail_reason))
                    continue

                valid_rows.append(row)

            if valid_rows:
                df_valid = pd.DataFrame(valid_rows)
                mode = "a" if os.path.exists(output_path) else "w"
                header = not os.path.exists(output_path)
                df_valid.to_csv(output_path, mode=mode, index=False, header=header)

            stats["kept"] += len(valid_rows)

            if (i + 1) % 10 == 0:
                print(
                    f"   ...已处理 {stats['total']} | 保留: {stats['kept']} ({stats['kept']/stats['total']:.1%})"
                )

        print("\n" + "=" * 50)
        print("✅ 清洗完成")
        print(f"📥 总数: {stats['total']}")
        print(f"💾 保留: {stats['kept']} ({stats['kept']/stats['total']*100:.2f}%)")
        print("-" * 30)
        print("🗑️ 剔除详情:")
        print(f"   ❌ 解析失败 (空数据/非法SMILES): {stats['err_parse']}")
        print(f"   👻 无中生有 (产物ID不在反应物): {stats['err_magic']}")
        print(f"   🧪 元素突变 (C变N等):           {stats['err_alchemy']}")
        print(
            f"   🌪️ 结构太乱 (变动原子 > {MAX_CHANGED_ATOMS}): {stats['err_scramble']}"
        )

        print("\n🔍 [调试] 前 5 个失败案例原因:")
        for idx, reason in debug_errors:
            print(f"   - Row {idx}: {reason}")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 严重错误: {e}")

if __name__ == "__main__":
    input_csv = "ord_data/ord_dataset_no_identical_products_20260313.csv"
    output_csv = "ord_data/ord_dataset_robust_mapped.csv"

    if os.path.exists(input_csv):
        robust_reaction_filter(input_csv, output_csv)
    else:
        print(f"找不到文件: {input_csv}")
