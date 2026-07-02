import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from torch_geometric.data import Data
import ast
from rdkit.Chem import rdFingerprintGenerator
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

INPUT_CSV = "./ord_data/23260313_3_ord_dataset_clean4.csv"
OUTPUT_DIR = "./ord_data/processed_ready_20260313"

CHUNK_SIZE = 64
MAX_ATOMS = 150
FP_SIZE = 1024
FP_RADIUS = 2

MFGEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_SIZE)

BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

def discretize_temperature(temp_val):
    """温度分桶策略 (8类)"""
    if pd.isna(temp_val) or temp_val == "":
        return 0
    try:
        t = float(temp_val)
    except:
        return 0
    if t <= -20:
        return 1
    if t <= 5:
        return 2
    if t < 20:
        return 3
    if t <= 35:
        return 4
    if t <= 70:
        return 5
    if t <= 120:
        return 6
    return 7

def get_atom_features(atom, is_aromatic_snapshot):
    """
    ✅ 修正：增加 is_aromatic_snapshot 参数
    严格对齐预训练模型的特征顺序
    """

    atomic_num = min(atom.GetAtomicNum(), 119)

    degree = min(atom.GetTotalDegree(), 14)

    charge = atom.GetFormalCharge()
    charge_idx = min(max(charge + 5, 0), 14)

    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
    }
    hyb = hyb_map.get(atom.GetHybridization(), 5)

    is_aromatic = 1 if is_aromatic_snapshot else 0

    mass = atom.GetMass()
    mass_idx = min(int(mass), 99)

    chirality = atom.GetChiralTag()
    chirality_idx = int(chirality)
    chirality_idx = min(chirality_idx, 9)

    num_hs = min(atom.GetTotalNumHs(), 9)

    return [
        atomic_num,
        degree,
        charge_idx,
        hyb,
        is_aromatic,
        mass_idx,
        chirality_idx,
        num_hs,
    ]

def delta_to_class(diff):
    """键级变化分类修正版"""
    diff = round(diff)
    if diff == 0:
        return 0
    elif diff == 1:
        return 1
    elif diff == -1:
        return 2
    elif diff == 2:
        return 3
    elif diff == -2:
        return 4
    elif diff == 3:
        return 5
    elif diff == -3:
        return 6

    if diff > 3:
        return 5
    if diff < -3:
        return 6
    return 0

def smiles_to_fp_list(env_str):
    if pd.isna(env_str) or not str(env_str).strip():
        return torch.zeros((1, FP_SIZE), dtype=torch.float)
    smiles_list = [s.strip() for s in str(env_str).split(";") if s.strip()]
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp_np = MFGEN.GetFingerprintAsNumPy(mol)
            fps.append(fp_np.astype(np.float32))
    if not fps:
        return torch.zeros((1, FP_SIZE), dtype=torch.float)
    return torch.tensor(np.array(fps), dtype=torch.float)

def process_single_row(row):
    try:

        r_smi_raw = row["Mapped_Reactants"]
        p_smi_raw = row["Mapped_Products"]
        if pd.isna(r_smi_raw) or pd.isna(p_smi_raw):
            return None

        r_smi = str(r_smi_raw).replace(" ", "")
        p_smi = str(p_smi_raw).replace(" ", "")

        r_mol = Chem.MolFromSmiles(r_smi)
        p_mol = Chem.MolFromSmiles(p_smi)

        if not r_mol or not p_mol:
            return None
        if r_mol.GetNumAtoms() > MAX_ATOMS:
            return None

        r_arom_atoms = set()
        for atom in r_mol.GetAtoms():
            if atom.GetIsAromatic():
                r_arom_atoms.add(atom.GetIdx())

        r_arom_bonds = set()
        for b in r_mol.GetBonds():
            if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                u, v = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                r_arom_bonds.add((u, v))

        p_arom_bonds = set()
        for b in p_mol.GetBonds():
            if b.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                u_idx, v_idx = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                atom_u = p_mol.GetAtomWithIdx(u_idx)
                atom_v = p_mol.GetAtomWithIdx(v_idx)
                map_u, map_v = atom_u.GetAtomMapNum(), atom_v.GetAtomMapNum()
                if map_u > 0 and map_v > 0:
                    p_arom_bonds.add(tuple(sorted((map_u, map_v))))

        try:
            Chem.SanitizeMol(r_mol)
            Chem.Kekulize(r_mol, clearAromaticFlags=True)
            Chem.SanitizeMol(p_mol)
            Chem.Kekulize(p_mol, clearAromaticFlags=True)
        except ValueError:
            return None

        atom_features = []
        for atom in r_mol.GetAtoms():

            is_arom_pre = atom.GetIdx() in r_arom_atoms
            atom_features.append(get_atom_features(atom, is_arom_pre))
        x = torch.tensor(atom_features, dtype=torch.long)

        src, dst, edge_attrs = [], [], []
        for bond in r_mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            if (u < v and (u, v) in r_arom_bonds) or (u > v and (v, u) in r_arom_bonds):
                b_type = BOND_TYPES[Chem.rdchem.BondType.AROMATIC]
            else:
                b_type = BOND_TYPES.get(bond.GetBondType(), 0)

            src.extend([u, v])
            dst.extend([v, u])
            edge_attrs.extend([b_type, b_type])

        if not src:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

        num_atoms = r_mol.GetNumAtoms()
        delta_matrix = np.zeros((num_atoms, num_atoms), dtype=np.int64)

        p_map = {
            a.GetAtomMapNum(): a.GetIdx()
            for a in p_mol.GetAtoms()
            if a.GetAtomMapNum() > 0
        }
        r_map = {
            a.GetIdx(): a.GetAtomMapNum()
            for a in r_mol.GetAtoms()
            if a.GetAtomMapNum() > 0
        }
        lg_labels = np.zeros(num_atoms, dtype=np.float32)

        for i in range(num_atoms):
            map_i = r_map.get(i)
            is_i_kept = map_i and map_i in p_map
            if not is_i_kept:
                lg_labels[i] = 1.0

            for j in range(i + 1, num_atoms):
                map_j = r_map.get(j)
                is_j_kept = map_j and map_j in p_map

                r_bond = r_mol.GetBondBetweenAtoms(i, j)
                r_type = r_bond.GetBondTypeAsDouble() if r_bond else 0.0
                p_type = 0.0

                if not is_i_kept and not is_j_kept:

                    delta_matrix[i, j] = 0
                    delta_matrix[j, i] = 0
                    continue

                elif (is_i_kept and not is_j_kept) or (not is_i_kept and is_j_kept):
                    p_type = 0.0

                else:
                    p_idx_i = p_map[map_i]
                    p_idx_j = p_map[map_j]
                    p_bond = p_mol.GetBondBetweenAtoms(p_idx_i, p_idx_j)
                    p_type = p_bond.GetBondTypeAsDouble() if p_bond else 0.0

                raw_diff = p_type - r_type

                is_noise = False
                if abs(raw_diff) == 1.0:
                    is_r_arom = (i, j) in r_arom_bonds
                    is_p_arom = False
                    if map_i and map_j:
                        is_p_arom = tuple(sorted((map_i, map_j))) in p_arom_bonds
                    if is_r_arom and is_p_arom:
                        is_noise = True

                if is_noise:
                    cls = 0
                else:
                    cls = delta_to_class(raw_diff)

                if cls != 0:
                    delta_matrix[i, j] = cls
                    delta_matrix[j, i] = cls

        env_features = smiles_to_fp_list(row["Environment_Molecules"])
        temp_id = discretize_temperature(row["Temperature"])
        try:
            time_val = float(row["Time"])
            if pd.isna(time_val):
                time_val = -1.0
        except:
            time_val = -1.0

        ab_flags = [
            int(row["Is_Strong_Acid"]),
            int(row["Is_Weak_Acid"]),
            int(row["Is_Strong_Base"]),
            int(row["Is_Weak_Base"]),
        ]

        proc_text = (
            str(row["Procedure_Text"]) if not pd.isna(row["Procedure_Text"]) else ""
        )
        work_text = str(row["Workup_Text"]) if not pd.isna(row["Workup_Text"]) else ""

        y_yield = (
            float(row["Yield"])
            if not pd.isna(row["Yield"]) and row["Yield"] != -1
            else 0.0
        )
        y_yield = max(0.0, min(100.0, y_yield)) / 100.0

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.y_delta = torch.tensor(delta_matrix, dtype=torch.long)
        data.y_lg = torch.tensor(lg_labels, dtype=torch.float)
        data.y_yield = torch.tensor([y_yield], dtype=torch.float)
        data.env_x = env_features.float()
        data.temp_id = torch.tensor([temp_id], dtype=torch.long)
        data.time_val = torch.tensor([time_val], dtype=torch.float)
        data.ab_flags = torch.tensor(ab_flags, dtype=torch.float).unsqueeze(0)
        data.proc_text = proc_text
        data.work_text = work_text
        data.smiles_r = r_smi
        data.smiles_p = p_smi
        data.rxn_id = row["Reaction ID"]
        data.num_nodes = num_atoms

        return data

    except Exception as e:
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    print(f"📂 Reading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["Mapped_Reactants", "Mapped_Products"])
    print(f"🚀 Processing {len(df)} samples...")

    buffer = []
    chunk_idx = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        data_obj = process_single_row(row)
        if data_obj:
            buffer.append(data_obj)
        if len(buffer) >= CHUNK_SIZE:
            save_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:06d}.pt")
            torch.save(buffer, save_path)
            buffer = []
            chunk_idx += 1
    if buffer:
        save_path = os.path.join(OUTPUT_DIR, f"chunk_{chunk_idx:06d}.pt")
        torch.save(buffer, save_path)
        chunk_idx += 1
    print(f"✅ Finished! Saved {chunk_idx} chunks to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
