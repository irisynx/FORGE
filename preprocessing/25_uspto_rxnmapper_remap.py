"""§73 USPTO RXNMapper 重映射 → ORD 原则切分 → 21_7 过滤 → PyG .pt chunks.

动机: USPTO-480k Jin 自带映射 = Lowe/Indigo automapper, 反应中心仅 87.3% 与 RXNMapper 一致,
      ~13% label noise (典型: 碱/试剂被异常拆解, 把产物原子错配到催化剂上, 见 rxn700221).
      用 RXNMapper (SOTA transformer mapper) 重新映射降噪。

处理原则 (对齐 ORD 23/21_7):
  1. 去掉原始映射 → RXNMapper 重新映射 reactants>>products
  2. ORD 切分: 反应物 frag 的 atom-map ∩ 产物 atom-map ≠ ∅ → 真反应物 (含 LG, Mapped_Reactants)
                                                = ∅ → 环境分子 (Environment_Molecules, 剥 map → Morgan FP)
  3. 21_7 校验: 产物每个 mapped 原子须在真反应物中存在同 map 同元素
                违反 (magic matter / 原数据漏反应物) → 删除
  4. 复用 23 的 process_single_row 算 y_delta

CLI: python 25_USPTO_RXNMapper重映射.py [split] [limit]
  split ∈ {train, valid, test, all};  limit: 仅前 N 行 (dryrun)
"""
import os
import sys
import importlib.util
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import torch
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

PROJECT_DIR = Path(__file__).parent.resolve()
ORD_SCRIPT = PROJECT_DIR / "23反应pt拆分原子特征.py"

INPUT_BASE = Path("./data/uspto_480k_jin/data")
OUTPUT_BASE = Path("./uspto_480k_rxnmapped")
SPLITS = ["train", "valid", "test"]
CHUNK_SIZE = 64
RXN_BATCH = 32
DEL_CONF_FLOOR = 0.10
MAX_BOND_CHANGES = 12

spec = importlib.util.spec_from_file_location("ord_pt", str(ORD_SCRIPT))
ord_pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ord_pt)
process_single_row = ord_pt.process_single_row

def strip_map_to_canonical(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)

def split_remapped(mapped_r_smi, mapped_p_smi):
    """ORD 原则切分 (用 RXNMapper 新 map): frag∩product → 真反应物, 否则 → 环境分子."""
    pm = Chem.MolFromSmiles(mapped_p_smi)
    if pm is None:
        return None
    p_maps = {a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum() > 0}
    if not p_maps:
        return None
    true_reac, reagents = [], []
    for frag in mapped_r_smi.split("."):
        m = Chem.MolFromSmiles(frag)
        if m is None:
            return None
        fmaps = {a.GetAtomMapNum() for a in m.GetAtoms() if a.GetAtomMapNum() > 0}
        if fmaps & p_maps:
            true_reac.append(frag)
        else:
            for a in m.GetAtoms():
                a.SetAtomMapNum(0)
            reagents.append(Chem.MolToSmiles(m))
    if not true_reac:
        return None
    return ".".join(true_reac), ";".join(reagents)

def validate_product_subset(mapped_reac, mapped_prod):
    """21_7 等价: 产物每个 mapped 原子须在反应物存在同 map 同元素. (ok, reason)"""
    rm = Chem.MolFromSmiles(mapped_reac)
    pm = Chem.MolFromSmiles(mapped_prod)
    if rm is None or pm is None:
        return False, "parse"
    r_map2z = {a.GetAtomMapNum(): a.GetAtomicNum() for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    for a in pm.GetAtoms():
        mp = a.GetAtomMapNum()
        if mp <= 0:
            continue
        if mp not in r_map2z:
            return False, "magic"
        if r_map2z[mp] != a.GetAtomicNum():
            return False, "alchemy"
    return True, ""

def _mapped_bonds(m):
    bs = set()
    for b in m.GetBonds():
        m1 = b.GetBeginAtom().GetAtomMapNum(); m2 = b.GetEndAtom().GetAtomMapNum()
        if m1 > 0 and m2 > 0:
            bs.add((min(m1, m2), max(m1, m2), int(b.GetBondTypeAsDouble())))
    return bs

def chem_checks_ok(mapped_reac, mapped_prod):
    """ORD 21_2 等价: 产物全重原子映射(除游离离子) + 元素守恒 + no magic + 键变化≤12."""
    rm = Chem.MolFromSmiles(mapped_reac); pm = Chem.MolFromSmiles(mapped_prod)
    if rm is None or pm is None:
        return False
    for a in pm.GetAtoms():
        if a.GetAtomicNum() <= 1:
            continue
        if a.GetAtomMapNum() == 0:
            if a.GetDegree() == 0 and a.GetFormalCharge() != 0:
                continue
            return False
    r_m2z = {a.GetAtomMapNum(): a.GetAtomicNum() for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    p_maps = {a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum() > 0}
    for a in pm.GetAtoms():
        mp = a.GetAtomMapNum()
        if mp <= 0:
            continue
        if mp not in r_m2z:
            return False
        if r_m2z[mp] != a.GetAtomicNum():
            return False
    common = set(r_m2z.keys()) & p_maps
    rb = {b for b in _mapped_bonds(rm) if b[0] in common and b[1] in common}
    pb = {b for b in _mapped_bonds(pm) if b[0] in common and b[1] in common}
    if len(rb - pb) + len(pb - rb) > MAX_BOND_CHANGES:
        return False
    return True

def _rc_sig(rsmi, psmi):
    """kept-kept 变化键 canonical 签名 (map-agnostic), 用于原映射 vs RXNMapper RC 对比."""
    rm = Chem.MolFromSmiles(rsmi); pm = Chem.MolFromSmiles(psmi)
    if rm is None or pm is None:
        return None
    r = {a.GetAtomMapNum(): a.GetIdx() for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    p = {a.GetAtomMapNum(): a.GetIdx() for a in pm.GetAtoms() if a.GetAtomMapNum() > 0}
    pm2 = Chem.Mol(pm)
    for a in pm2.GetAtoms():
        a.SetAtomMapNum(0)
    rank = list(Chem.CanonicalRankAtoms(pm2))
    sig = set(); ks = list(p.keys())
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            mi, mj = ks[i], ks[j]
            if mi not in r or mj not in r:
                continue
            rb = rm.GetBondBetweenAtoms(r[mi], r[mj]); pb = pm.GetBondBetweenAtoms(p[mi], p[mj])
            rbt = rb.GetBondTypeAsDouble() if rb else 0.0; pbt = pb.GetBondTypeAsDouble() if pb else 0.0
            if abs(rbt - pbt) > 0.4:
                sig.add((min(rank[p[mi]], rank[p[mj]]), max(rank[p[mi]], rank[p[mj]]), round(pbt - rbt, 1)))
    return frozenset(sig)

def build_unmapped_rxn(raw_rxn):
    """USPTO 行 → 去 map 的 'reactants>>products' (喂 RXNMapper)."""
    try:
        rs, _, ps = raw_rxn.split(">")
    except ValueError:
        return None
    rs_c = strip_map_to_canonical(rs)
    ps_c = strip_map_to_canonical(ps)
    if not rs_c or not ps_c:
        return None
    return rs_c + ">>" + ps_c

def convert_split(split, rxn_mapper, limit=None):
    in_file = INPUT_BASE / f"{split}.txt"
    out_dir = OUTPUT_BASE / split
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 split={split} in={in_file} out={out_dir}/ limit={limit or 'ALL'}")

    stats = dict(total=0, ok=0, skip_parse=0, skip_rxnmap=0, skip_disagree_lowconf=0,
                 skip_chem=0, skip_magic=0, skip_split=0, skip_pt=0)
    buffer, chunk_idx = [], 0

    raw_lines = []
    raw_map = {}
    with open(in_file) as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            ls = line.strip()
            raw_lines.append((idx, ls))
            raw_map[idx] = ls.split()[0] if ls else ""

    pbar = tqdm(range(0, len(raw_lines), RXN_BATCH), desc=split)
    for bstart in pbar:
        batch = raw_lines[bstart:bstart + RXN_BATCH]
        unmapped, meta = [], []
        for idx, line in batch:
            stats["total"] += 1
            raw_rxn = line.split()[0] if line else ""
            unm = build_unmapped_rxn(raw_rxn)
            if unm is None:
                stats["skip_parse"] += 1
                continue
            unmapped.append(unm)
            meta.append(idx)
        if not unmapped:
            continue
        try:
            results = rxn_mapper.get_attention_guided_atom_maps(unmapped)
        except Exception:

            results = []
            for u in unmapped:
                try:
                    results.append(rxn_mapper.get_attention_guided_atom_maps([u])[0])
                except Exception:
                    results.append(None)
        for idx, res in zip(meta, results):
            if res is None:
                stats["skip_rxnmap"] += 1
                continue
            conf = res.get("confidence", 0.0)
            mapped_rxn = res["mapped_rxn"]
            try:
                mr, _, mp = mapped_rxn.split(">")
            except ValueError:
                stats["skip_rxnmap"] += 1
                continue
            sp = split_remapped(mr, mp)
            if sp is None:
                stats["skip_split"] += 1
                continue
            mapped_reac, env_smi = sp

            if not chem_checks_ok(mapped_reac, mp):
                stats["skip_chem"] += 1
                continue
            ok_map, reason = validate_product_subset(mapped_reac, mp)
            if not ok_map:
                stats["skip_magic"] += 1
                continue

            if conf < DEL_CONF_FLOOR:
                raw_full = raw_map.get(idx, "")
                try:
                    o_rs, _, o_ps = raw_full.split(">")
                    if _rc_sig(o_rs, o_ps) != _rc_sig(mapped_reac, mp):
                        stats["skip_disagree_lowconf"] += 1
                        continue
                except Exception:
                    pass
            row = {
                "Mapped_Reactants": mapped_reac,
                "Mapped_Products": mp,
                "Environment_Molecules": env_smi,
                "Temperature": "", "Time": "",
                "Is_Strong_Acid": 0, "Is_Weak_Acid": 0,
                "Is_Strong_Base": 0, "Is_Weak_Base": 0,
                "Procedure_Text": "", "Workup_Text": "",
                "Yield": -1,
                "Reaction ID": f"uspto_rxnmap_{split}_{idx}",
            }
            data_obj = process_single_row(row)
            if data_obj is None:
                stats["skip_pt"] += 1
                continue
            data_obj.rxnmap_conf = torch.tensor([float(conf)], dtype=torch.float)
            buffer.append(data_obj)
            stats["ok"] += 1
            if len(buffer) >= CHUNK_SIZE:
                torch.save(buffer, out_dir / f"chunk_{chunk_idx:06d}.pt")
                buffer = []
                chunk_idx += 1
        pbar.set_postfix(ok=stats["ok"], magic=stats["skip_magic"], rxnf=stats["skip_rxnmap"])

    if buffer:
        torch.save(buffer, out_dir / f"chunk_{chunk_idx:06d}.pt")
        chunk_idx += 1
    print(f"✅ {split}: {stats} chunks={chunk_idx}")
    return stats, chunk_idx

def main():
    split_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    splits = SPLITS if split_arg == "all" else [split_arg]

    print("加载 RXNMapper...")
    from rxnmapper import RXNMapper
    rxn_mapper = RXNMapper()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary = []
    for split in splits:
        st, nc = convert_split(split, rxn_mapper, limit=limit)
        summary.append((split, st, nc))

    print("\n=== Summary ===")
    for s, st, nc in summary:
        kept_rate = 100 * st["ok"] / st["total"] if st["total"] else 0
        magic_rate = 100 * st["skip_magic"] / st["total"] if st["total"] else 0
        print(f"{s}: total={st['total']} ok={st['ok']} ({kept_rate:.1f}%) "
              f"magic_del={st['skip_magic']} ({magic_rate:.1f}%) "
              f"rxnmap_fail={st['skip_rxnmap']} chunks={nc}")

if __name__ == "__main__":
    main()
