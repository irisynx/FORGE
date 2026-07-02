import os
import subprocess
import torch
import numpy as np
from tqdm import tqdm
import tempfile
import shutil
from rdkit import Chem

DATA_DIR = "/data/chara/project/AI4S/data/processed_data"

XTB_BIN = shutil.which("xtb") or "/home/chara/xtb-dist/bin/xtb"

def parse_xtb_fukui(output_text, num_atoms):
    """
    解析 xTB Log 获取 Fukui 指数
    兼容: '1N 0.1 0.2' (粘连) 和 '1 N 0.1 0.2' (标准)
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

def parse_xtb_wbo_smart(wbo_file_path, num_atoms):
    """解析 WBO 文件 (稀疏或稠密)"""
    if not os.path.exists(wbo_file_path):
        return None
    try:
        raw = np.loadtxt(wbo_file_path)

        if raw.ndim == 2 and raw.shape[1] == 3:
            mat = np.zeros((num_atoms, num_atoms))
            for r in raw:
                i, j, v = int(r[0]) - 1, int(r[1]) - 1, r[2]
                if 0 <= i < num_atoms and 0 <= j < num_atoms:
                    mat[i, j] = mat[j, i] = v
            return torch.tensor(mat, dtype=torch.float)

        elif raw.shape == (num_atoms, num_atoms):
            return torch.tensor(raw, dtype=torch.float)
        else:
            return None
    except:
        return None

def parse_xtb_charges_file(charge_file_path, num_atoms):
    """
    直接读取 charges 文件
    返回形状: [num_atoms] (一维 Tensor)
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

def run_xtb_repair(data, file_path):
    """运行 xTB 并更新 Fukui, WBO, Charge"""
    try:
        if data.x.dim() == 2:
            atoms = data.x[:, 0].tolist()
        else:
            atoms = data.x.tolist()
        atoms = [int(a) for a in atoms]
    except:
        return False

    if hasattr(data, "pos_target"):
        coords = data.pos_target.tolist()
    else:
        coords = data.pos.tolist()

    num_atoms = len(atoms)
    pt = Chem.GetPeriodicTable()
    try:
        symbols = [pt.GetElementSymbol(z) for z in atoms]
    except:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        xyz_file = os.path.join(tmpdir, "mol.xyz")
        with open(xyz_file, "w") as f:
            f.write(f"{num_atoms}\n\n")
            for s, c in zip(symbols, coords):
                f.write(f"{s} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

        cmd = f"{XTB_BIN} mol.xyz --gfn 2 --vfukui --wbo --chrg 0"

        env = os.environ.copy()
        xtb_home = os.path.dirname(os.path.dirname(XTB_BIN))
        env["XTBPATH"] = os.path.join(xtb_home, "share/xtb")

        try:
            result = subprocess.run(
                cmd, shell=True, cwd=tmpdir, capture_output=True, text=True, env=env
            )

            if result.returncode != 0:
                return False

            fukui_tensor = parse_xtb_fukui(result.stdout, num_atoms)

            wbo_path = os.path.join(tmpdir, "wbo")
            wbo_tensor = parse_xtb_wbo_smart(wbo_path, num_atoms)

            charge_path = os.path.join(tmpdir, "charges")
            charge_tensor = parse_xtb_charges_file(charge_path, num_atoms)

            if wbo_tensor is None or charge_tensor is None:
                return False

            data.y_fukui = fukui_tensor
            data.y_wbo = wbo_tensor
            data.y_charge = charge_tensor

            torch.save(data, file_path)
            return True

        except Exception:
            return False

def main():
    print(f"🚀 Starting Full Repair (Fukui + WBO + Charge)")
    print(f"📂 Data Directory: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print("Error: Directory not found.")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pt")]

    fixed_count = 0
    failed_count = 0

    pbar = tqdm(files)
    for f in pbar:
        full_path = os.path.join(DATA_DIR, f)

        try:
            data = torch.load(full_path, weights_only=False)

            success = run_xtb_repair(data, full_path)

            if success:
                fixed_count += 1
                pbar.set_description(f"Updated: {fixed_count}")
            else:
                failed_count += 1

        except Exception:
            failed_count += 1

    print("\n" + "=" * 30)
    print("🎉 All Done!")
    print(f"✅ Updated: {fixed_count}")
    print(f"❌ Failed:  {failed_count}")

if __name__ == "__main__":
    main()
