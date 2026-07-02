import os
import shutil
import subprocess
import numpy as np
import torch
import re
from tqdm import tqdm
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger

INPUT_FILE = "data/Properties_20250319.txt"
OUTPUT_DIR = "./data/processed_ready_20260131"

XTB_BIN = shutil.which("xtb") or "/home/chara/xtb-dist/bin/xtb"

RDLogger.DisableLog("rdApp.*")

def parse_raw_block(block_lines):
    """
    解析文本块，提取坐标和元数据。
    (逻辑源自您的 00 代码)
    """
    smiles_ref = "unknown"
    symbols = []
    coords = []

    for line in block_lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("smiles ="):
            try:
                smiles_ref = line.split("=")[1].strip().split()[0]
            except:
                pass
            continue

        if "=" in line:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

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
    (逻辑源自您的 00 代码)
    """
    num_atoms = len(symbols)
    xyz_block = f"{num_atoms}\n\n"
    for s, c in zip(symbols, coords):
        xyz_block += f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n"

    try:

        raw_mol = Chem.MolFromXYZBlock(xyz_block)
        if raw_mol is None:
            return None

        mol = Chem.Mol(raw_mol)

        try:
            rdDetermineBonds.DetermineBonds(mol, charge=0)
        except:

            try:
                mol = Chem.Mol(raw_mol)
                rdDetermineBonds.DetermineBonds(mol)
            except:
                return None

        return mol
    except:
        return None

def get_atom_features(atom):

    atom_num = atom.GetAtomicNum()
    degree = atom.GetTotalDegree()
    formal_charge = atom.GetFormalCharge() + 2
    hybridization = int(atom.GetHybridization())
    if hybridization > 6:
        hybridization = 6
    is_aromatic = int(atom.GetIsAromatic())
    mass = int(atom.GetMass() / 5)
    chirality = 0
    total_h = atom.GetTotalNumHs()
    return torch.tensor(
        [
            atom_num,
            degree,
            formal_charge,
            hybridization,
            is_aromatic,
            mass,
            chirality,
            total_h,
        ],
        dtype=torch.long,
    )

def get_edge_features(bond):
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        feat = 0
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        feat = 1
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        feat = 2
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        feat = 3
    else:
        feat = 4
    return feat

def parse_xtb_fukui(output_text, num_atoms):
    """
    解析 xTB Log 获取 Fukui 指数 (完全照搬您的 01 代码)
    """
    fukui = np.zeros((num_atoms, 2))
    lines = output_text.splitlines()
    start_reading = False
    idx = 0

    for line in lines:
        if "Fukui index" in line:
            start_reading = False
        if "f(+)" in line and "f(-)" in line:
            start_reading = True
            continue

        if start_reading:
            parts = line.strip().split()
            if not parts:
                continue
            if "----" in line or "====" in line:
                if idx > 0:
                    start_reading = False
                continue

            if idx < num_atoms:
                try:
                    col0 = parts[0]

                    if not col0.isdigit() and col0[0].isdigit():
                        vals = [float(parts[1]), float(parts[2])]

                    elif col0.isdigit() and parts[1][0].isalpha():
                        vals = [float(parts[2]), float(parts[3])]

                    elif col0.isdigit():
                        vals = [float(parts[1]), float(parts[2])]
                    else:
                        continue

                    fukui[idx] = vals
                    idx += 1
                except:
                    pass
    return torch.tensor(fukui, dtype=torch.float)

def parse_xtb_energy(output_text):
    """
    (修正版) 解析 HOMO/LUMO/Gap
    日志格式示例:
    16        2.0000           -0.4260408             -11.5932 (HOMO)
    17                         -0.2942287              -8.0064 (LUMO)
    HL-Gap            0.1318121 Eh            3.5868 eV
    """
    homo, lumo, gap = None, None, None
    lines = output_text.splitlines()
    for line in lines:
        parts = line.split()
        if not parts:
            continue

        if parts[-1] == "(HOMO)":
            try:
                homo = float(parts[-2])
            except:
                pass

        elif parts[-1] == "(LUMO)":
            try:
                lumo = float(parts[-2])
            except:
                pass

        elif "HL-Gap" in line and "eV" in line:
            try:

                idx = parts.index("eV")
                gap = float(parts[idx - 1])
            except:
                pass

    if gap is None and homo is not None and lumo is not None:
        gap = lumo - homo

    return homo, lumo, gap

def parse_xtb_charges_file(charge_file_path, num_atoms):
    """
    (完全照搬 01 代码)
    """
    if not os.path.exists(charge_file_path):
        return None
    try:
        data = np.loadtxt(charge_file_path)
        if data.shape[0] != num_atoms:
            return None
        return torch.tensor(data, dtype=torch.float)
    except:
        return None

def run_xtb_calc(coords, symbols):
    """
    运行 xTB (整合逻辑)
    """
    num_atoms = len(symbols)

    cwd = os.getcwd()
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:

            xyz_file = "mol.xyz"
            with open(xyz_file, "w") as f:
                f.write(f"{num_atoms}\n\n")
                for s, c in zip(symbols, coords):
                    f.write(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

            cmd = [XTB_BIN, xyz_file, "--vfukui", "--gfn", "2"]

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"

            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, env=env, timeout=120
            )

            output = result.stdout

            fukui = parse_xtb_fukui(output, num_atoms)
            charges = parse_xtb_charges_file("charges", num_atoms)
            homo, lumo, gap = parse_xtb_energy(output)

            if charges is None:
                charges = torch.zeros(num_atoms)

            if homo is None:
                homo = 0.0
            if lumo is None:
                lumo = 0.0
            if gap is None:
                gap = 0.0

            return {
                "fukui": fukui,
                "charge": charges,
                "homo": homo,
                "lumo": lumo,
                "gap": gap,
            }

        except Exception as e:

            return None
        finally:
            os.chdir(cwd)

def process_single_block(block_lines, file_idx):

    smiles_ref, symbols, coords = parse_raw_block(block_lines)
    if not symbols:
        return False

    mol = build_mol_from_xyz(symbols, coords)
    if mol is None:
        return False

    x_list = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(x_list)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        attr = get_edge_features(bond)
        edge_indices += [[i, j], [j, i]]
        edge_attrs += [attr, attr]

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)

    pos = torch.tensor(coords, dtype=torch.float)

    xtb_res = run_xtb_calc(coords, symbols)
    if xtb_res is None:
        return False

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        y_fukui=xtb_res["fukui"],
        y_charge=xtb_res["charge"],
        homo=torch.tensor([xtb_res["homo"]], dtype=torch.float),
        lumo=torch.tensor([xtb_res["lumo"]], dtype=torch.float),
        gap=torch.tensor([xtb_res["gap"]], dtype=torch.float),
        smiles=smiles_ref,
    )

    torch.save(data, os.path.join(OUTPUT_DIR, f"mol_{file_idx}.pt"))
    return True

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_size = os.path.getsize(INPUT_FILE)

    print(f"Processing {INPUT_FILE} ({total_size / (1024**3):.2f} GB)...")

    current_block = []
    mol_count = 0
    success_count = 0

    with tqdm(total=total_size, unit="B", unit_scale=True, desc="Progress") as pbar:
        with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:

                pbar.update(len(line.encode("utf-8")))

                line_str = line.strip()
                if not line_str:
                    continue

                if line_str.startswith("smiles ="):

                    if current_block:
                        if process_single_block(current_block, mol_count):
                            success_count += 1
                        mol_count += 1

                        if mol_count % 10 == 0:
                            pbar.set_postfix(mol=mol_count, success=success_count)

                        current_block = []
                    current_block.append(line_str)
                else:
                    current_block.append(line_str)

            if current_block:
                if process_single_block(current_block, mol_count):
                    success_count += 1
                mol_count += 1
                pbar.set_postfix(mol=mol_count, success=success_count)

    print(f"\nProcessing Complete!")
    print(f"Total Molecules: {mol_count}")
    print(f"Successfully Saved: {success_count}")
    print(f"Output Directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
