"""§73.2 RXNMapper 重映射审计可视化 (人在环路).

召回"原 USPTO 映射 ≠ RXNMapper 重映射"的反应 (双映射器分歧 → 至少一个错),
按反应类型 + confidence 分桶, 画 [原映射 RC | RXNMapper 映射 RC | 产物] 三联对比卡片,
供人工判定: 原映射对 / RXNMapper 对 / 都不合理(删除).

CLI: python 26_remap_audit_viz.py [n_sample] [top_per_bucket]
输出: analysis_reports/remap_audit/{cards/*.png, summary.txt, index.html}
"""
import os, sys, random, warnings
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
RDLogger.DisableLog("rdApp.*")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

PROJECT = Path(__file__).parent.resolve()
OUT = PROJECT / "analysis_reports" / "remap_audit"
(OUT / "cards").mkdir(parents=True, exist_ok=True)

RC_CLASS_RULES = [
    ("C-N", lambda zs: 6 in zs and 7 in zs),
    ("C-O", lambda zs: 6 in zs and 8 in zs),
    ("C-Halogen", lambda zs: 6 in zs and (9 in zs or 17 in zs or 35 in zs or 53 in zs)),
    ("C-S", lambda zs: 6 in zs and 16 in zs),
    ("C-C", lambda zs: zs == {6}),
]

def rc_atoms_and_sig(rsmi, psmi):
    """返回 (reactant RC 原子 idx 集合, kept-kept 变化键 canonical 签名, RC 原子元素集)."""
    rm = Chem.MolFromSmiles(rsmi); pm = Chem.MolFromSmiles(psmi)
    if rm is None or pm is None:
        return None, None, None
    r_m2i = {a.GetAtomMapNum(): a.GetIdx() for a in rm.GetAtoms() if a.GetAtomMapNum() > 0}
    p_m2i = {a.GetAtomMapNum(): a.GetIdx() for a in pm.GetAtoms() if a.GetAtomMapNum() > 0}
    pm2 = Chem.Mol(pm)
    for a in pm2.GetAtoms():
        a.SetAtomMapNum(0)
    rank = list(Chem.CanonicalRankAtoms(pm2))
    rc_atoms = set(); sig = set(); zs = set()
    ks = list(p_m2i.keys())

    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            mi, mj = ks[i], ks[j]
            if mi not in r_m2i or mj not in r_m2i:
                continue
            rb = rm.GetBondBetweenAtoms(r_m2i[mi], r_m2i[mj])
            pb = pm.GetBondBetweenAtoms(p_m2i[mi], p_m2i[mj])
            rbt = rb.GetBondTypeAsDouble() if rb else 0.0
            pbt = pb.GetBondTypeAsDouble() if pb else 0.0
            if abs(rbt - pbt) > 0.4:
                sig.add((min(rank[p_m2i[mi]], rank[p_m2i[mj]]), max(rank[p_m2i[mi]], rank[p_m2i[mj]]), round(pbt - rbt, 1)))
                rc_atoms.add(r_m2i[mi]); rc_atoms.add(r_m2i[mj])
                zs.add(rm.GetAtomWithIdx(r_m2i[mi]).GetAtomicNum())
                zs.add(rm.GetAtomWithIdx(r_m2i[mj]).GetAtomicNum())

    for b in rm.GetBonds():
        ai, bi = b.GetBeginAtom(), b.GetEndAtom()
        ma, mb = ai.GetAtomMapNum(), bi.GetAtomMapNum()
        in_a = ma in p_m2i; in_b = mb in p_m2i
        if in_a != in_b:
            rc_atoms.add(ai.GetIdx()); rc_atoms.add(bi.GetIdx())
            zs.add(ai.GetAtomicNum()); zs.add(bi.GetAtomicNum())
    return rc_atoms, frozenset(sig), zs

def classify(zs):
    for name, fn in RC_CLASS_RULES:
        try:
            if fn(zs):
                return name
        except Exception:
            pass
    return "other"

def draw_mol_rc(smi, rc_atoms, title):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(420, 340)
    hl = [i for i in (rc_atoms or []) if i < m.GetNumAtoms()]
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=hl,
                                       highlightAtomColors={i: (1, 0.5, 0.5) for i in hl})
    d.FinishDrawing()
    return Image.open(BytesIO(d.GetDrawingText()))

def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    top_per = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    lines = open("data/uspto_480k_jin/data/test.txt").read().splitlines()
    random.seed(42); random.shuffle(lines)
    unm, orig = [], []
    for ln in lines:
        if len(unm) >= n_sample:
            break
        try:
            rxn = ln.split()[0]; rs, _, ps = rxn.split(">")
            rm = Chem.MolFromSmiles(rs); pm = Chem.MolFromSmiles(ps)
            if not rm or not pm:
                continue
            for a in rm.GetAtoms(): a.SetAtomMapNum(0)
            for a in pm.GetAtoms(): a.SetAtomMapNum(0)
            unm.append(Chem.MolToSmiles(rm) + ">>" + Chem.MolToSmiles(pm))
            orig.append((rs, ps))
        except Exception:
            pass

    print(f"加载 RXNMapper, 重映射 {len(unm)} 反应...")
    from rxnmapper import RXNMapper
    rmap = RXNMapper()
    res = []
    for i in range(0, len(unm), 32):
        try:
            res.extend(rmap.get_attention_guided_atom_maps(unm[i:i + 32]))
        except Exception:
            res.extend([None] * len(unm[i:i + 32]))

    agree = 0; disagree = []
    for idx, r in enumerate(res):
        if r is None:
            continue
        conf = r["confidence"]
        try:
            o_atoms, o_sig, o_zs = rc_atoms_and_sig(orig[idx][0], orig[idx][1])
            mr, _, mp = r["mapped_rxn"].split(">")
            x_atoms, x_sig, x_zs = rc_atoms_and_sig(mr, mp)
        except Exception:
            continue
        if o_sig is None or x_sig is None:
            continue
        if o_sig == x_sig:
            agree += 1
        else:
            cls = classify(o_zs | x_zs)
            disagree.append(dict(idx=idx, conf=conf, cls=cls,
                                 orig_r=orig[idx][0], orig_p=orig[idx][1], orig_atoms=o_atoms,
                                 rx_r=mr, rx_p=mp, rx_atoms=x_atoms))

    tot = agree + len(disagree)

    buckets = defaultdict(list)
    for d in disagree:
        buckets[d["cls"]].append(d)
    for k in buckets:
        buckets[k].sort(key=lambda d: d["conf"])

    lines_out = [f"=== RXNMapper 重映射审计 (test {len(res)} 抽样) ===",
                 f"AGREE (原映射==RXNMapper): {agree} ({100*agree/tot:.1f}%)",
                 f"DISAGREE (双映射器分歧): {len(disagree)} ({100*len(disagree)/tot:.1f}%)",
                 "", "DISAGREE 按反应类型分桶 (低 conf 优先):"]
    for k in sorted(buckets, key=lambda k: -len(buckets[k])):
        cs = [d["conf"] for d in buckets[k]]
        lines_out.append(f"  {k:12s}: {len(buckets[k]):4d}  conf均值={sum(cs)/len(cs):.3f}  低conf(<0.3)={sum(c<0.3 for c in cs)}")
    summary = "\n".join(lines_out)
    print(summary)
    (OUT / "summary.txt").write_text(summary)

    n_card = 0; html = ["<html><body><h2>RXNMapper 重映射审计 — 双映射器分歧反应</h2><pre>" + summary + "</pre>"]
    for cls in sorted(buckets, key=lambda k: -len(buckets[k])):
        html.append(f"<h3>{cls} (n={len(buckets[cls])})</h3>")
        for d in buckets[cls][:top_per]:
            fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
            for k, (smi, atoms, ttl) in enumerate([
                (d["orig_r"], d["orig_atoms"], "原 USPTO 映射 (RC 红)"),
                (d["rx_r"], d["rx_atoms"], "RXNMapper 映射 (RC 红)"),
                (d["rx_p"], None, "产物")]):
                img = draw_mol_rc(smi, atoms, ttl)
                if img: ax[k].imshow(img)
                ax[k].set_title(ttl, fontsize=9); ax[k].axis("off")
            fig.suptitle(f"[{cls}] idx={d['idx']} conf={d['conf']:.3f}  (原RC {len(d['orig_atoms'])}原子 vs RXNMapper {len(d['rx_atoms'])}原子)", fontsize=10)
            fn = OUT / "cards" / f"{cls}__{d['idx']:05d}_conf{int(d['conf']*100):03d}.png"
            fig.savefig(fn, dpi=90, bbox_inches="tight"); plt.close(fig)
            html.append(f'<img src="cards/{fn.name}" width="900"><br>')
            n_card += 1
    (OUT / "index.html").write_text("\n".join(html))
    print(f"\n渲染 {n_card} 张对比卡片 → {OUT}/cards/  + index.html")

if __name__ == "__main__":
    main()
