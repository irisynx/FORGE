"""§73.3 RXNMapper 重映射删除阈值细分桶统计 (定阈值用).

删除标准 (A方案): 删 = DISAGREE(原映射RC≠RXNMapper RC) ∩ conf < 阈值.
AGREE 的无论 conf 多低都保留 (两 mapper 同意=可信).

输出: confidence 细分桶 × AGREE/DISAGREE + 各阈值删除统计.
CLI: python 27_remap_threshold_stats.py [n_sample]  (默认全量 test)
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

def rc_sig(rsmi, psmi):
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

def main():
    import random
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
    lines = open("data/uspto_480k_jin/data/test.txt").read().splitlines()
    if n_sample < len(lines):
        random.seed(42); random.shuffle(lines); lines = lines[:n_sample]
    unm, orig = [], []
    for ln in lines:
        try:
            rxn = ln.split()[0]; rs, _, ps = rxn.split(">")
            rm = Chem.MolFromSmiles(rs); pm = Chem.MolFromSmiles(ps)
            if not rm or not pm:
                continue
            for a in rm.GetAtoms(): a.SetAtomMapNum(0)
            for a in pm.GetAtoms(): a.SetAtomMapNum(0)
            unm.append(Chem.MolToSmiles(rm) + ">>" + Chem.MolToSmiles(pm)); orig.append((rs, ps))
        except Exception:
            pass
    print(f"重映射 {len(unm)} 反应...")
    from rxnmapper import RXNMapper
    rmap = RXNMapper()
    rows = []
    for i in range(0, len(unm), 32):
        try:
            res = rmap.get_attention_guided_atom_maps(unm[i:i + 32])
        except Exception:
            res = [None] * len(unm[i:i + 32])
        for k, r in enumerate(res):
            if r is None:
                continue
            try:
                o = rc_sig(orig[i + k][0], orig[i + k][1])
                mr, _, mp = r["mapped_rxn"].split(">")
                x = rc_sig(mr, mp)
            except Exception:
                continue
            if o is None or x is None:
                continue
            rows.append((r["confidence"], o == x))
        if (i // 32) % 50 == 0:
            print(f"  {i}/{len(unm)}", flush=True)

    N = len(rows)
    edges = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.01]
    print(f"\n=== confidence 细分桶 (test n={N}) ===")
    print(f"{'桶':14s}{'total':>7}{'占%':>7}{'AGREE':>7}{'DISAGR':>8}{'DISAGR率':>9}")
    for a, b in zip(edges, edges[1:]):
        sub = [ag for c, ag in rows if a <= c < b]
        if not sub:
            continue
        dis = sum(1 for x in sub if not x)
        print(f"[{a:.2f},{b:.2f}){'':3s}{len(sub):>7}{100*len(sub)/N:>6.1f}%{sum(sub):>7}{dis:>8}{100*dis/len(sub):>8.1f}%")

    total_dis = sum(1 for _, ag in rows if not ag)
    print(f"\n总 DISAGREE: {total_dis} ({100*total_dis/N:.1f}%)")
    print(f"\n=== A方案删除统计 (删 = DISAGREE ∩ conf<阈值; AGREE 全保留) ===")
    print(f"{'阈值T':>7}{'删除数':>8}{'删占总%':>9}{'占DISAGREE%':>13}{'删除集中DISAGREE纯度':>22}")
    for T in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:

        del_set = [(c, ag) for c, ag in rows if c < T and not ag]

        below = [(c, ag) for c, ag in rows if c < T]
        purity = 100 * sum(1 for _, ag in below if not ag) / len(below) if below else 0
        ndel = len(del_set)
        print(f"{T:>7.2f}{ndel:>8}{100*ndel/N:>8.1f}%{100*ndel/total_dis:>12.1f}%{purity:>20.1f}%")
    print("\n说明: '删占总%'=删多少数据; '占DISAGREE%'=召回多少分歧反应;")
    print("      '纯度'= conf<T 区间里 DISAGREE 占比 (但删除只删其中DISAGREE, AGREE保留, 故无误删)")

if __name__ == "__main__":
    main()
