import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import os

INPUT_CSV = "./ord_data/ord_processed_samples_refined_cleaned_dataset.csv"
OUTPUT_CSV = "./ord_data/ord_processed_samples_refined_cleaned_dataset_sane.csv"
ERROR_CSV = "./ord_data/ord_rejected_samples.csv"

MAX_BOND_CHANGES = 12

RDLogger.DisableLog("rdApp.*")

class SanityChecker:
    def __init__(self):
        self.stats = {
            "total": 0,
            "kept": 0,
            "rejected_atom_mismatch": 0,
            "rejected_too_complex": 0,
            "rejected_parse_error": 0,
            "rejected_unmapped_product": 0,
            "count_zero_changes": 0,
        }

    def get_mapped_bonds(self, mol):
        bonds = set()
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            map1 = a1.GetAtomMapNum()
            map2 = a2.GetAtomMapNum()
            if map1 > 0 and map2 > 0:
                u, v = sorted((map1, map2))
                b_order = int(bond.GetBondTypeAsDouble())
                bonds.add((u, v, b_order))
        return bonds

    def check_full_product_mapping(self, p_mol):
        """
        检查产物的所有重原子是否都有映射。
        忽略：氢原子、游离离子(无键且带电)。
        """
        for atom in p_mol.GetAtoms():

            if atom.GetAtomicNum() <= 1:
                continue

            if atom.GetAtomMapNum() == 0:

                if atom.GetDegree() == 0 and atom.GetFormalCharge() != 0:
                    continue

                return False
        return True

    def check_reaction(self, r_smiles, p_smiles):
        r_mol = Chem.MolFromSmiles(r_smiles)
        p_mol = Chem.MolFromSmiles(p_smiles)

        if r_mol is None or p_mol is None:
            return False, "parse_error", 0

        if not self.check_full_product_mapping(p_mol):
            return False, "unmapped_product", 0

        r_atoms = {}
        for atom in r_mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                r_atoms[atom.GetAtomMapNum()] = atom.GetAtomicNum()

        p_map_ids = set()
        for atom in p_mol.GetAtoms():
            pid = atom.GetAtomMapNum()
            if pid > 0:
                p_map_ids.add(pid)
                if pid in r_atoms:
                    if atom.GetAtomicNum() != r_atoms[pid]:
                        return False, "atom_mismatch", 0

        r_bonds = self.get_mapped_bonds(r_mol)
        p_bonds = self.get_mapped_bonds(p_mol)

        common_ids = set(r_atoms.keys()).intersection(p_map_ids)

        def filter_bonds(bonds, valid_ids):
            return {b for b in bonds if b[0] in valid_ids and b[1] in valid_ids}

        r_bonds_core = filter_bonds(r_bonds, common_ids)
        p_bonds_core = filter_bonds(p_bonds, common_ids)

        broken = r_bonds_core - p_bonds_core
        formed = p_bonds_core - r_bonds_core
        num_changes = len(broken) + len(formed)

        if num_changes == 0:
            self.stats["count_zero_changes"] += 1

        if num_changes > MAX_BOND_CHANGES:
            return False, "too_complex", num_changes

        return True, "ok", num_changes

def main():
    print(f"🚀 开始执行映射神智检查 (严格产物映射模式)...")
    print(f"📂 输入: {INPUT_CSV}")

    if not os.path.exists(INPUT_CSV):
        print("❌ 输入文件不存在！")
        return

    checker = SanityChecker()
    chunk_size = 100000

    try:
        total_rows = sum(1 for _ in open(INPUT_CSV)) - 1
    except:
        total_rows = 0

    header = True
    reader = pd.read_csv(INPUT_CSV, chunksize=chunk_size)

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    if os.path.exists(ERROR_CSV):
        os.remove(ERROR_CSV)

    pbar = tqdm(total=total_rows, unit="rxn")

    for chunk in reader:
        valid_rows = []
        rejected_rows = []

        for idx, row in chunk.iterrows():
            checker.stats["total"] += 1

            r_smi = row.get("Mapped_Reactants", "")
            p_smi = row.get("Mapped_Products", "")

            if pd.isna(r_smi) or pd.isna(p_smi):
                checker.stats["rejected_parse_error"] += 1
                continue

            is_valid, reason, changes = checker.check_reaction(r_smi, p_smi)

            row["Calc_Changes"] = changes

            if is_valid:
                checker.stats["kept"] += 1
                valid_rows.append(row)
            else:
                checker.stats[f"rejected_{reason}"] += 1
                row["Reject_Reason"] = reason
                rejected_rows.append(row)

        if valid_rows:
            pd.DataFrame(valid_rows).to_csv(
                OUTPUT_CSV, mode="a", header=header, index=False
            )
        if rejected_rows:
            pd.DataFrame(rejected_rows).to_csv(
                ERROR_CSV,
                mode="a" if os.path.exists(ERROR_CSV) else "w",
                header=header,
                index=False,
            )

        header = False
        pbar.update(len(chunk))

    pbar.close()

    s = checker.stats
    total = max(s["total"], 1)
    kept = s["kept"]

    print(f"\n{'='*20} 📊 最终检查报告 {'='*20}")
    print(f"📥 输入总数: {total}")
    print(f"✅ 保留通过: {kept} ({kept/total:.1%})")
    print(
        f"   ℹ️ 其中无变化反应(Zero Changes): {s['count_zero_changes']} (占通过数据的 {s['count_zero_changes']/max(kept,1):.1%})"
    )
    print(f"❌ 拒绝总数: {total - kept}")
    print(f"   ├─ 产物原子未映射: {s['rejected_unmapped_product']} (严格模式)")
    print(f"   ├─ 元素突变 (幻觉): {s['rejected_atom_mismatch']}")
    print(f"   ├─ 过于复杂 (>12变化): {s['rejected_too_complex']}")
    print(f"   └─ 解析错误: {s['rejected_parse_error']}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
