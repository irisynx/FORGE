"""USPTO-480k (Jin rexgen) → PyG .pt chunks.

输入: data/uspto_480k_jin/data/{train,valid,test}.txt
  每行格式: <reactants_mixed>>><products> <bond_changes>
  - reactants_mixed: 真反应物 + 试剂/溶剂混合 (全原子带 [X:N] mapping)
  - products: 产物 SMILES (全原子带 mapping, 100% 子集于 reactants mapping)
  - bond_changes: a-b;a-b;... (atom-map id 对, 此字段不使用, y_delta 由 RDKit 重算)

输出: uspto_480k_processed/{train,valid,test}/chunk_*.pt (每 chunk 64 反应)
  字段对齐 ORD pipeline (与 23反应pt拆分原子特征.py 完全一致), 缺失条件占位.

切分规则 (frag-level):
  frag.atom_maps ∩ product.atom_maps ≠ ∅  → 真反应物 (含 LG), 拼 Mapped_Reactants
  frag.atom_maps ∩ product.atom_maps = ∅   → 试剂/溶剂, 拼 Environment_Molecules

占位条件 (USPTO Jin 不带):
  Temperature="", Time="", ab_flags=[0,0,0,0], Procedure_Text="", Workup_Text="",
  Yield=-1, Reaction ID="uspto_jin_{split}_{idx}"
"""

import os
import sys
import importlib.util
from pathlib import Path

import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_DIR = Path(__file__).parent.resolve()
ORD_SCRIPT = PROJECT_DIR / "23反应pt拆分原子特征.py"

INPUT_BASE = Path("./data/uspto_480k_jin/data")
OUTPUT_BASE = Path("./uspto_480k_processed")
SPLITS = ["train", "valid", "test"]

CHUNK_SIZE = 64

spec = importlib.util.spec_from_file_location("ord_pt", str(ORD_SCRIPT))
ord_pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ord_pt)
process_single_row = ord_pt.process_single_row

def split_reactants_and_reagents(reactants_mixed_smi: str, product_smi: str):
    """按 frag mapping 与 product mapping 是否相交切分.

    Returns:
        mapped_reactants_smi: '.' 连真反应物 frag (保留 atom-mapping, 含 LG)
        env_smi_list_str: ';' 连试剂 SMILES (剥 atom-mapping 后)
        ok: False 表示解析失败 → 跳过
    """

    p_mol = Chem.MolFromSmiles(product_smi)
    if p_mol is None:
        return "", "", False
    product_maps = {a.GetAtomMapNum() for a in p_mol.GetAtoms() if a.GetAtomMapNum() > 0}
    if not product_maps:
        return "", "", False

    reactant_frags = reactants_mixed_smi.split(".")
    true_reactant_smis = []
    reagent_smis_stripped = []
    for smi in reactant_frags:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return "", "", False
        frag_maps = {a.GetAtomMapNum() for a in m.GetAtoms() if a.GetAtomMapNum() > 0}
        if frag_maps & product_maps:
            true_reactant_smis.append(smi)
        else:

            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            reagent_smis_stripped.append(Chem.MolToSmiles(m))

    if not true_reactant_smis:
        return "", "", False

    return (
        ".".join(true_reactant_smis),
        ";".join(reagent_smis_stripped),
        True,
    )

def validate_product_subset(mapped_reac: str, mapped_prod: str):
    """对齐 ORD 21_7映射检查.py 的 no-magic-matter + no-alchemy 语义.

    产物每个 mapped 原子必须在反应物中存在同 map 且同元素.
    Returns: (ok, reason). reason in {"", "parse", "magic", "alchemy"}.
    """
    rm = Chem.MolFromSmiles(mapped_reac)
    pm = Chem.MolFromSmiles(mapped_prod)
    if rm is None or pm is None:
        return False, "parse"
    r_map2z = {a.GetAtomMapNum(): a.GetAtomicNum()
               for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    for a in pm.GetAtoms():
        mp = a.GetAtomMapNum()
        if mp <= 0:
            continue
        if mp not in r_map2z:
            return False, "magic"
        if r_map2z[mp] != a.GetAtomicNum():
            return False, "alchemy"
    return True, ""

def uspto_line_to_row(line: str, split: str, idx: int):
    """USPTO txt 行 → 23 脚本 process_single_row() 期待的 row dict."""
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    rxn = parts[0]
    try:
        reactants_mixed, _, product = rxn.split(">")
    except ValueError:
        return None
    if not reactants_mixed or not product:
        return None

    mapped_reac, env_smi_list, ok = split_reactants_and_reagents(reactants_mixed, product)
    if not ok:
        return None

    return {
        "Mapped_Reactants": mapped_reac,
        "Mapped_Products": product,
        "Environment_Molecules": env_smi_list,
        "Temperature": "",
        "Time": "",
        "Is_Strong_Acid": 0,
        "Is_Weak_Acid": 0,
        "Is_Strong_Base": 0,
        "Is_Weak_Base": 0,
        "Procedure_Text": "",
        "Workup_Text": "",
        "Yield": -1,
        "Reaction ID": f"uspto_jin_{split}_{idx}",
    }

def convert_split(split: str, limit: int = None):
    in_file = INPUT_BASE / f"{split}.txt"
    out_dir = OUTPUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 split={split}  in={in_file}  out={out_dir}/  limit={limit or 'ALL'}")

    n_total = 0
    n_ok = 0
    n_skip_split = 0
    n_skip_pt = 0
    n_skip_mapcheck = 0
    buffer = []
    chunk_idx = 0

    with open(in_file) as f:
        pbar = tqdm(f, desc=f"{split}")
        for idx, line in enumerate(pbar):
            if limit and n_total >= limit:
                break
            n_total += 1
            row = uspto_line_to_row(line, split, idx)
            if row is None:
                n_skip_split += 1
                continue

            ok_map, _reason = validate_product_subset(
                row["Mapped_Reactants"], row["Mapped_Products"])
            if not ok_map:
                n_skip_mapcheck += 1
                continue
            data_obj = process_single_row(row)
            if data_obj is None:
                n_skip_pt += 1
                continue
            buffer.append(data_obj)
            n_ok += 1
            if len(buffer) >= CHUNK_SIZE:
                torch.save(buffer, out_dir / f"chunk_{chunk_idx:06d}.pt")
                buffer = []
                chunk_idx += 1
            if n_total % 5000 == 0:
                pbar.set_postfix(ok=n_ok, skip_split=n_skip_split, skip_pt=n_skip_pt, chunks=chunk_idx)
    if buffer:
        torch.save(buffer, out_dir / f"chunk_{chunk_idx:06d}.pt")
        chunk_idx += 1

    print(f"✅ {split}: total={n_total} ok={n_ok} skip_split={n_skip_split} skip_pt={n_skip_pt} chunks={chunk_idx}")
    return n_total, n_ok, n_skip_split, n_skip_pt, chunk_idx

def main():

    split_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    splits = SPLITS if split_arg == "all" else [split_arg]

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in splits:
        summary.append((split, *convert_split(split, limit=limit)))

    print("\n=== Summary ===")
    print(f"{'split':<8}{'total':>10}{'ok':>10}{'skip_split':>14}{'skip_pt':>10}{'chunks':>10}")
    for s, t, o, ss, sp, c in summary:
        print(f"{s:<8}{t:>10}{o:>10}{ss:>14}{sp:>10}{c:>10}")

if __name__ == "__main__":
    main()
