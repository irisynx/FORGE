"""§73.4 化学守恒检验有效性验证: 原 USPTO 映射 vs RXNMapper 重映射.

借鉴 ORD 21_2/21_4/21_7 化学检验:
  ① 产物全重原子映射 (除游离抗衡离子 Degree=0+带电)
  ② 元素守恒 (mapped 同 AtomicNum)
  ③ no magic matter (产物 map ⊆ 反应物 map)
  ④ 键变化数 ≤ 12 (broken+formed, common_ids)

对比原映射 vs RXNMapper 各项通过率 + 键变化数分布, 标记被拦反应供可视化.
CLI: python 28_chem_check_validation.py [n_sample]
"""
import sys, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

MAX_BOND_CHANGES = 12

def mapped_bonds(m):
    bs = set()
    for b in m.GetBonds():
        m1 = b.GetBeginAtom().GetAtomMapNum(); m2 = b.GetEndAtom().GetAtomMapNum()
        if m1 > 0 and m2 > 0:
            bs.add((min(m1, m2), max(m1, m2), int(b.GetBondTypeAsDouble())))
    return bs

def chem_checks(rsmi, psmi):
    rm = Chem.MolFromSmiles(rsmi); pm = Chem.MolFromSmiles(psmi)
    if rm is None or pm is None:
        return None

    full_map = True
    for a in pm.GetAtoms():
        if a.GetAtomicNum() <= 1:
            continue
        if a.GetAtomMapNum() == 0:
            if a.GetDegree() == 0 and a.GetFormalCharge() != 0:
                continue
            full_map = False
            break
    r_m2z = {a.GetAtomMapNum(): a.GetAtomicNum() for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    p_maps = {a.GetAtomMapNum() for a in pm.GetAtoms() if a.GetAtomMapNum() > 0}

    elem_ok = True; no_magic = True
    for a in pm.GetAtoms():
        mp = a.GetAtomMapNum()
        if mp <= 0:
            continue
        if mp not in r_m2z:
            no_magic = False
        elif r_m2z[mp] != a.GetAtomicNum():
            elem_ok = False

    common = set(r_m2z.keys()) & p_maps
    rb = {b for b in mapped_bonds(rm) if b[0] in common and b[1] in common}
    pb = {b for b in mapped_bonds(pm) if b[0] in common and b[1] in common}
    n_changes = len(rb - pb) + len(pb - rb)
    all_pass = full_map and elem_ok and no_magic and (n_changes <= MAX_BOND_CHANGES)
    return dict(full_map=full_map, elem_ok=elem_ok, no_magic=no_magic,
                n_changes=n_changes, all_pass=all_pass)

def main():
    import random
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    lines = open("data/uspto_480k_jin/data/test.txt").read().splitlines()
    random.seed(42); random.shuffle(lines)
    unm, orig = [], []
    for ln in lines:
        if len(unm) >= n:
            break
        try:
            rxn = ln.split()[0]; rs, _, ps = rxn.split(">")
            rm = Chem.MolFromSmiles(rs); pm = Chem.MolFromSmiles(ps)
            if not rm or not pm:
                continue
            rs_keep, ps_keep = rs, ps
            for a in rm.GetAtoms(): a.SetAtomMapNum(0)
            for a in pm.GetAtoms(): a.SetAtomMapNum(0)
            unm.append(Chem.MolToSmiles(rm) + ">>" + Chem.MolToSmiles(pm))
            orig.append((rs_keep, ps_keep))
        except Exception:
            pass

    print(f"重映射 {len(unm)} 反应...", flush=True)
    from rxnmapper import RXNMapper
    rmap = RXNMapper()

    orig_stat = dict(full=0, elem=0, magic=0, bc12=0, allpass=0, nbc=[])
    rx_stat = dict(full=0, elem=0, magic=0, bc12=0, allpass=0, nbc=[])
    cnt = 0
    rx_blocked = []
    orig_blocked_bc = 0
    rx_blocked_bc = 0
    for i in range(0, len(unm), 32):
        try:
            res = rmap.get_attention_guided_atom_maps(unm[i:i + 32])
        except Exception:
            res = [None] * len(unm[i:i + 32])
        for k, r in enumerate(res):
            oc = chem_checks(orig[i + k][0], orig[i + k][1])
            if oc is None:
                continue
            cnt += 1
            orig_stat["full"] += oc["full_map"]; orig_stat["elem"] += oc["elem_ok"]
            orig_stat["magic"] += oc["no_magic"]; orig_stat["bc12"] += (oc["n_changes"] <= 12)
            orig_stat["allpass"] += oc["all_pass"]; orig_stat["nbc"].append(oc["n_changes"])
            if oc["n_changes"] > 12:
                orig_blocked_bc += 1
            if r is None:
                continue
            try:
                mr, _, mp = r["mapped_rxn"].split(">")
            except ValueError:
                continue
            xc = chem_checks(mr, mp)
            if xc is None:
                continue
            rx_stat["full"] += xc["full_map"]; rx_stat["elem"] += xc["elem_ok"]
            rx_stat["magic"] += xc["no_magic"]; rx_stat["bc12"] += (xc["n_changes"] <= 12)
            rx_stat["allpass"] += xc["all_pass"]; rx_stat["nbc"].append(xc["n_changes"])
            if xc["n_changes"] > 12:
                rx_blocked_bc += 1
            if not xc["all_pass"] and len(rx_blocked) < 40:
                rx_blocked.append(dict(idx=i + k, conf=round(r["confidence"], 3),
                                       reason=("bc>12" if xc["n_changes"] > 12 else
                                               "no_full_map" if not xc["full_map"] else
                                               "elem" if not xc["elem_ok"] else "magic"),
                                       n_changes=xc["n_changes"],
                                       mapped_rxn=r["mapped_rxn"][:400]))

    def pct(x): return 100 * x / cnt if cnt else 0
    import statistics as st
    print(f"\n=== 化学检验通过率: 原 USPTO 映射 vs RXNMapper 重映射 (n={cnt}) ===")
    print(f"{'检验':22s}{'原映射':>10}{'RXNMapper':>12}")
    print(f"{'① 产物全重原子映射':20s}{pct(orig_stat['full']):>9.1f}%{pct(rx_stat['full']):>11.1f}%")
    print(f"{'② 元素守恒':24s}{pct(orig_stat['elem']):>9.1f}%{pct(rx_stat['elem']):>11.1f}%")
    print(f"{'③ no magic matter':23s}{pct(orig_stat['magic']):>9.1f}%{pct(rx_stat['magic']):>11.1f}%")
    print(f"{'④ 键变化数≤12':22s}{pct(orig_stat['bc12']):>9.1f}%{pct(rx_stat['bc12']):>11.1f}%")
    print(f"{'★ 全部通过':24s}{pct(orig_stat['allpass']):>9.1f}%{pct(rx_stat['allpass']):>11.1f}%")
    print(f"\n键变化数: 原映射 mean={st.mean(orig_stat['nbc']):.2f} max={max(orig_stat['nbc'])} | "
          f"RXNMapper mean={st.mean(rx_stat['nbc']):.2f} max={max(rx_stat['nbc'])}")
    print(f"键变化数 >12 (碱拆解类信号): 原映射 {orig_blocked_bc} ({pct(orig_blocked_bc):.1f}%) | "
          f"RXNMapper {rx_blocked_bc} ({pct(rx_blocked_bc):.1f}%)")
    print(f"\nRXNMapper 全部通过 = 重映射后干净保留率")
    print(f"RXNMapper 化学检验未过 (删除候选): {cnt - rx_stat['allpass']} ({pct(cnt-rx_stat['allpass']):.1f}%)")

    out = Path("analysis_reports/chem_check_blocked.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rx_blocked, ensure_ascii=False, indent=2))
    print(f"\nRXNMapper 被拦反应样本 → {out} ({len(rx_blocked)} 个)")

if __name__ == "__main__":
    main()
