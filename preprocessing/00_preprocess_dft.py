import os
import subprocess
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from tqdm import tqdm
import io

INPUT_FILE = "data/Properties_20250319.txt"
OUTPUT_DIR = "./processed_data"
XTB_BIN = "xtb"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_raw_block(block_lines):
    """
    解析文本块，提取坐标和元数据。
    """
    smiles_ref = "unknown"
    symbols = []
    coords = []

    for line in block_lines:
        line = line.strip()
        if not line: continue

        if line.startswith("smiles ="):
            try:
                smiles_ref = line.split("=")[1].strip().split()[0]
            except:
                pass
            continue

        if "=" in line: continue

        parts = line.split()
        if len(parts) < 5: continue

        sym = parts[0]

        if sym[0].isalpha() and len(sym) <= 2:
            try:

                _ = Chem.GetPeriodicTable().GetAtomicNumber(sym)
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                symbols.append(sym)
                coords.append([x, y, z])
            except:
                continue

    return smiles_ref, symbols, np.array(coords)

def build_mol_from_xyz(symbols, coords):
    """
    核心黑科技：从 XYZ 坐标直接重建 RDKit 分子对象。
    """
    num_atoms = len(symbols)
    xyz_block = f"{num_atoms}\n\n"
    for s, c in zip(symbols, coords):
        xyz_block += f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"

    try:

        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        if raw_mol is None: return None

        mol = Chem.Mol(raw_mol)

        rdDetermineBonds.DetermineBonds(mol, charge=0)

        return mol
    except Exception as e:

        return None

def process_stream(file_path):
    current_block = []
    mol_counter = 0
    success_count = 0

    with open(file_path, 'r') as f:
        for line in tqdm(f, desc="Processing"):
            if line.startswith("smiles =") and current_block:
                if process_single_molecule(current_block, mol_counter):
                    success_count += 1
                current_block = [line]
                mol_counter += 1
            else:
                current_block.append(line)

        if current_block:
            if process_single_molecule(current_block, mol_counter):
                success_count += 1

    print(f"\n成功处理分子数: {success_count}/{mol_counter}")

def process_single_molecule(block_lines, mol_id):
    smiles_ref, xyz_symbols, xyz_coords = parse_raw_block(block_lines)

    if len(xyz_coords) == 0: return False

    mol = build_mol_from_xyz(xyz_symbols, xyz_coords)

    if mol is None:

        return False

    atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    x = torch.tensor(atomic_nums, dtype=torch.long).unsqueeze(1)

    adj = Chem.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long)

    pos_target = torch.tensor(xyz_coords, dtype=torch.float)

    fukui = torch.zeros((len(xyz_symbols), 2))

    data = Data(x=x, edge_index=edge_index)
    data.pos_target = pos_target
    data.y_fukui = fukui
    data.smiles_ref = smiles_ref

    torch.save(data, os.path.join(OUTPUT_DIR, f"mol_{mol_id}.pt"))
    return True

if __name__ == "__main__":
    print("启动 XYZ 重建流水线...")
    process_stream(INPUT_FILE)
