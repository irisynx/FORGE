import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm
import os

INPUT_CSV = "./ord_data/ord_processed_samples_refined_cleaned_dataset_sane_features.csv"

OUTPUT_CSV = (
    "./ord_data/ord_processed_samples_refined_cleaned_dataset_sane_features_mapped.csv"
)

REJECT_CSV = "./ord_data/ord_dataset_strict_rejected.csv"

CHUNK_SIZE = 50000

RDLogger.DisableLog("rdApp.*")

CHUNK_SIZE = 50000

RDLogger.DisableLog("rdApp.*")

class RobustMappingValidator:
    def __init__(self):
        self.stats = {
            "total": 0,
            "passed": 0,
            "rejected_parse_error": 0,
            "rejected_unmapped_atom": 0,
            "rejected_ghost_mapping": 0,
            "rejected_empty": 0,
        }

    def extract_map_ids(self, smiles_str):
        """
        鲁棒地提取 SMILES 中所有的 Atom Map ID。
        策略：按 '.' 分割，逐个解析，防止因单个碎片错误导致整体失败。
        """
        if pd.isna(smiles_str) or str(smiles_str).strip() == "":
            return set()

        map_ids = set()

        fragments = str(smiles_str).split(".")

        for frag in fragments:
            frag = frag.strip()
            if not frag:
                continue

            mol = Chem.MolFromSmiles(frag)
            if mol:
                for atom in mol.GetAtoms():
                    mid = atom.GetAtomMapNum()
                    if mid > 0:
                        map_ids.add(mid)
        return map_ids

    def validate_row(self, r_smiles, p_smiles):

        if pd.isna(r_smiles) or pd.isna(p_smiles):
            return False, "empty"

        reactant_ids = self.extract_map_ids(r_smiles)

        if not reactant_ids:

            pass

        p_smiles_clean = str(p_smiles).replace(" . ", ".").replace(" ", "")
        p_mol = Chem.MolFromSmiles(p_smiles_clean)

        if not p_mol:
            return False, "parse_error"

        for atom in p_mol.GetAtoms():

            if atom.GetAtomicNum() == 1:
                continue

            if atom.GetDegree() == 0 and atom.GetFormalCharge() != 0:
                continue

            mid = atom.GetAtomMapNum()

            if mid == 0:
                return False, "unmapped_atom"

            if mid not in reactant_ids:
                return False, "ghost_mapping"

        return True, "passed"

def main():
    print(f"🕵️‍♂️ 启动鲁棒性映射检查程序...")
    print(f"📂 输入: {INPUT_CSV}")

    if not os.path.exists(INPUT_CSV):
        print("❌ 错误: 输入文件不存在")
        return

    validator = RobustMappingValidator()

    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
    if os.path.exists(REJECT_CSV):
        os.remove(REJECT_CSV)

    try:
        total_rows = sum(1 for _ in open(INPUT_CSV)) - 1
    except:
        total_rows = 0

    reader = pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)
    header = True

    pbar = tqdm(total=total_rows, unit="rxn")

    for chunk in reader:
        valid_rows = []
        reject_rows = []

        for idx, row in chunk.iterrows():
            validator.stats["total"] += 1

            r_smi = row.get("Mapped_Reactants", "")
            p_smi = row.get("Mapped_Products", "")

            is_valid, reason = validator.validate_row(r_smi, p_smi)

            if is_valid:
                validator.stats["passed"] += 1
                valid_rows.append(row)
            else:
                validator.stats[f"rejected_{reason}"] += 1
                row["Reject_Reason"] = reason
                reject_rows.append(row)

        if valid_rows:
            pd.DataFrame(valid_rows).to_csv(
                OUTPUT_CSV, mode="a", header=header, index=False
            )
        if reject_rows:
            pd.DataFrame(reject_rows).to_csv(
                REJECT_CSV, mode="a", header=header, index=False
            )

        header = False
        pbar.update(len(chunk))

    pbar.close()

    s = validator.stats
    total = max(s["total"], 1)

    print(f"\n{'='*20} 📊 检查报告 {'='*20}")
    print(f"📥 总数: {s['total']}")
    print(f"✅ 通过: {s['passed']} ({s['passed']/total:.1%})")
    print(f"❌ 拒绝: {total - s['passed']}")
    print(f"   ├─ 产物原子未映射 (Map=0): {s['rejected_unmapped_atom']}")
    print(f"   ├─ 幻觉映射 (ID不在反应物中): {s['rejected_ghost_mapping']}")
    print(f"   └─ 格式/解析错误: {s['rejected_parse_error']}")
    print(f"{'='*50}")
    print(f"💾 结果: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
