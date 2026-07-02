import os
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

DATA_DIR = "./data/processed_data"

def get_atom_features(atom):
    """
    提取原子的 8 维特征:
    [AtomicNum, Degree, NumHs, Valence, Hybridization, FormalCharge, IsAromatic, IsInRing]
    """

    atomic_num = min(atom.GetAtomicNum(), 99)

    degree = min(atom.GetTotalDegree(), 9)

    num_hs = min(atom.GetTotalNumHs(), 9)

    valence = min(atom.GetExplicitValence(), 9)

    hyb_map = {
        Chem.rdchem.HybridizationType.SP: 0,
        Chem.rdchem.HybridizationType.SP2: 1,
        Chem.rdchem.HybridizationType.SP3: 2,
        Chem.rdchem.HybridizationType.SP3D: 3,
        Chem.rdchem.HybridizationType.SP3D2: 4,
        Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
    }
    hyb = hyb_map.get(atom.GetHybridization(), 5)

    charge = min(max(atom.GetFormalCharge() + 5, 0), 9)

    is_aromatic = 1 if atom.GetIsAromatic() else 0

    is_in_ring = 1 if atom.IsInRing() else 0

    return [atomic_num, degree, num_hs, valence, hyb, charge, is_aromatic, is_in_ring]

def get_bond_type_from_wbo(wbo_val):
    """
    根据 WBO 值判断键类型
    0:Single, 1:Double, 2:Triple, 3:Aromatic
    """
    if wbo_val < 1.3:
        return 0
    elif 1.3 <= wbo_val < 1.8:
        return 3
    elif 1.8 <= wbo_val < 2.5:
        return 1
    else:
        return 2

def build_mol_from_data(data):
    """
    尝试从 Data 对象中重建 RDKit Mol 对象。
    策略 A: 使用 smiles_ref (最快，最准)
    策略 B: 使用 pos_target + atomic_nums (兜底，适用于 SMILES 缺失情况)
    """
    mol = None

    if (
        hasattr(data, "smiles_ref")
        and isinstance(data.smiles_ref, str)
        and data.smiles_ref != "unknown"
    ):
        try:
            mol = Chem.MolFromSmiles(data.smiles_ref)

            if mol and mol.GetNumAtoms() == data.num_nodes:
                return mol

            if mol:
                mol = Chem.AddHs(mol)
                if mol.GetNumAtoms() == data.num_nodes:
                    return mol
        except:
            pass

    if hasattr(data, "pos_target") and data.x is not None:
        try:

            if data.x.dim() > 1:
                atomic_nums = data.x[:, 0].tolist()
            else:
                atomic_nums = data.x.tolist()

            coords = data.pos_target.tolist()
            symbols = [
                Chem.GetPeriodicTable().GetElementSymbol(int(z)) for z in atomic_nums
            ]

            num_atoms = len(symbols)
            xyz_block = f"{num_atoms}\n\n"
            for s, c in zip(symbols, coords):
                xyz_block += f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"

            raw_mol = Chem.MolFromXYZBlock(xyz_block)
            if raw_mol:
                mol = Chem.Mol(raw_mol)
                rdDetermineBonds.DetermineBonds(mol, charge=0)
                if mol.GetNumAtoms() == data.num_nodes:
                    return mol
        except Exception as e:
            pass

    return None

def process_file(file_path):
    try:
        data = torch.load(file_path, weights_only=False)

        mol = build_mol_from_data(data)
        if mol is None:
            print(
                f"Skipping {os.path.basename(file_path)}: Cannot reconstruct molecule."
            )
            return False

        new_features = []
        for atom in mol.GetAtoms():
            new_features.append(get_atom_features(atom))

        data.x = torch.tensor(new_features, dtype=torch.long)

        if hasattr(data, "y_wbo") and hasattr(data, "edge_index"):
            edge_index = data.edge_index
            y_wbo = data.y_wbo
            num_edges = edge_index.size(1)
            edge_attrs = []

            for k in range(num_edges):
                i = edge_index[0, k].item()
                j = edge_index[1, k].item()

                if i < y_wbo.shape[0] and j < y_wbo.shape[1]:
                    wbo_val = y_wbo[i, j].item()
                else:
                    wbo_val = 0.0

                b_type = get_bond_type_from_wbo(wbo_val)
                edge_attrs.append(b_type)

            data.edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

        torch.save(data, file_path)
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pt")]
    print(f"Found {len(files)} files in {DATA_DIR}. Starting comprehensive update...")

    success_count = 0
    for f in tqdm(files):
        path = os.path.join(DATA_DIR, f)
        if process_file(path):
            success_count += 1

    print(f"\nDone! Updated {success_count}/{len(files)} files.")
    print("Updates applied:")
    print(
        "1. data.x -> [N, 8] (AtomicNum, Degree, NumHs, Valence, Hybridization, Charge, Aromatic, Ring)"
    )
    print("2. data.edge_attr -> Calculated from y_wbo (if available)")

if __name__ == "__main__":
    main()
