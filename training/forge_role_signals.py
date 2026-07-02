"""v25 路线 A: 43 维 SMARTS 信号提取 (v25.1 拆 has_C_eq_O→4)

按 EditGNN_Architecture.md §48.2 设计:
  A.1 偶联与配体     (8 维) — Pd/Ni/Cu/Pt/phosphine/boron/zinc/tin
  A.2 试剂角色       (9 维) — dehydrating/oxidant/reductant/coupling/Lewis A/Lewis B/PT/protecting/radical
  A.3 官能团 + ab    (16 维) — NH/OH/SH/alkyne/alkene + 4 carbonyl 子(aldehyde/ketone/ester_amide/anhydride_acyl_X)
                              + aryl_X/COOH/aldehyde_only + ab×4
  A.4 结构 motif     (10 维) — epoxide/aziridine_oxetane/cyclopropane/β-lactam/allyl_LG/benzyl_LG/1,5-diene/D-A pair/o-aryl-LG/spiro

合计 43 维 (v25.0 = 40, v25.1 拆 has_C_eq_O 72.6%→4 子信号 提升细分度).
温度/时间不在此 43 维内 (走 v_cond 单独路径).

使用:
  from forge_role_signals import extract_role_signals, extract_role_signals_batch
  feat_b = extract_role_signals(smiles_r, env_smiles_list, proc_text, work_text, ab_flags)  # [40]
  feat_B = extract_role_signals_batch(...)                                                   # [B, 40]
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Union

import torch
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

A1_DEFS = [
    ("has_metal_Pd",       ["[Pd]"],                                     [],                                  "env"),
    ("has_metal_Ni",       ["[Ni]"],                                     [],                                  "env"),
    ("has_metal_Cu",       ["[Cu]"],                                     [],                                  "env"),
    ("has_metal_Pt",       ["[Pt]"],                                     [],                                  "env"),
    ("has_phosphine",      ["[PX3](-[#6])(-[#6])-[#6]",
                            "[PX3](-c)(-c)-c"],                          [],                                  "env"),
    ("has_boron",          ["[#6][B]([OH])[OH]",
                            "[#6][B]1OC(=O)C(=O)O1",
                            "[#6][B]([OX2][CX4])([OX2][CX4])",
                            "[#6][B]([CX4])[CX4]"],                      [],                                  "both"),
    ("has_zinc_organyl",   ["[#6][Zn]"],                                 [],                                  "both"),
    ("has_tin_organyl",    ["[Sn]-[#6]"],                                [],                                  "both"),
]

A2_DEFS = [
    ("has_dehydrating",   [
        "[H]OS(=O)(=O)O[H]",
        "OS(=O)(=O)O",
        "O=P(O)(O)OP(=O)(O)O",
        "O=P(Cl)(Cl)Cl",
        "O=P(O)(O)Cl",
        "O=S(Cl)Cl",
        "FC(F)(F)S(=O)(=O)OS(=O)(=O)C(F)(F)F",
        "[Mg+2].[O-]S([O-])(=O)=O",
        "[Ca+2].[H-].[H-]",
        "N=C=NC1CCCCC1",
    ],
    ["dean-stark", "dean stark", "molecular sieves", "4a sieves", "4 a sieves",
     "4å sieves", "molecular sieve", "molsieve", "dehydrat",
     "mgso4", "na2so4", "dried over", "anhydrous", "drying agent"],
    "env"),

    ("has_oxidant",       [

        "O=C(OO)c1ccccc1Cl",
        "[O-][Mn](=O)(=O)=O",
        "O=[Cr](=O)=O",
        "N#CC1=C(C#N)C(=O)C(Cl)=C(Cl)C1=O",
        "O=I(=O)(O)O",
        "[O]N1CCCCC1",
        "O=[Mn]=O",
        "O=[Se]=O",
    ],

    ["mcpba", "kmno4", "cro3", "cr o3", "dmp", "dess-martin", "dess martin",
     "ddq", "ibx", "pcc", "swern", "oxone", "tempo", "naio4", "mn o2", "mno2",
     "selenium dioxide", "seo2", "jones reagent", "ipy2bf4"],
    "env"),

    ("has_reductant",     [

        "[Na+].[BH4-]",
        "[Li+].[AlH4-]",
        "[Na+].[BH3-]C#N",
        "[Na+].[BH-](OC(=O)C)(OC(=O)C)OC(=O)C",
    ],

    ["nabh4", "sodium borohydride", "naborohydride",
     "lialh4", "lithium aluminum hydride", "lithium aluminium",
     "dibal", "dibal-h", "diisobutylaluminum",
     "borane", "thf-borane", "bh3·thf", "bh3-thf", "bh3.smr2", "bh3.thf", "bh3 thf",
     "pd/c", "pd-c", "pt/c", "pt-c", "rh/c", "ru/c",
     "h2/pd", "h2 / pd", "h2/pt", "h2/raney",
     "h2 atmos", "h2 pressure", "hydrogenation", "hydrogenated",
     "raney ni", "raney-ni", "raney nickel", "nicl2/nabh4",
     "nacnbh3", "sodium cyanoborohydride",
     "nabh(oac)3", "na(oac)3bh", "stab", "sodium triacetoxyborohydride",
     "stryker", "lindlar", "wilkinson",
     "et3sih", "(tms)3sih", "phsih3",
     "samarium iodide", "smi2", "sm i2",
     "l-selectride", "k-selectride",
     "tributyltin hydride", "bu3snh"],
    "env"),

    ("has_coupling_reagent", [
        "N=C=NC1CCCCC1",
        "CCN=C=NCCCN(C)C",
        "O=C(n1ccnc1)n2ccnc2",

    ],
    ["dcc", "edc", "edci", "edc.hcl", "hbtu", "hatu", "pybop", "pybroP",
     "tbtu", "bop", "t3p", "cdi", "carbonyldiimidazole", "dicyclohexylcarbodiimide"],
    "env"),

    ("has_lewis_acid",    [
        "B(F)(F)F",
        "Cl[Al](Cl)Cl",
        "Cl[Ti](Cl)(Cl)Cl",
        "Cl[Zn]Cl",
        "Cl[Fe](Cl)Cl",
        "[Sc+3]",
        "[Yb+3]",
        "Cl[In](Cl)Cl",
        "B(c1ccc(F)cc1)(c1ccc(F)cc1)c1ccc(F)cc1",
    ],
    ["bf3", "alcl3", "tibu4", "ticl4", "zncl2", "fecl3", "sc(otf)3",
     "yb(otf)3", "incl3", "cu(otf)2", "scotf3", "ybotf3", "lewis acid"],
    "env"),

    ("has_lewis_base",    [
        "c1ccnc(N(C)C)c1",
        "CCN(CC)CC",
        "CCN(C(C)C)C(C)C",
        "c1ccncc1",
        "c1cnc[nH]1",
        "N1=CN(C)c2ncnc12",
    ],
    ["dmap", "dbu", "tea", "et3n", "triethylamine", "hunig", "hunig's",
     "ipr2net", "diisopropylethylamine", "pyridine", "imidazole", "nmm",
     "n-methylmorpholine", "lutidine", "collidine", "n-methylimidazole"],
    "env"),

    ("has_phase_transfer", [
        "CCCC[N+](CCCC)(CCCC)CCCC",
        "CCCCCCCC[N+](C)(CCCCCCCC)CCCCCCCC",
        "O1CCOCCOCCOCCOCCOCC1",
        "O1CCOCCOCCOCC1",
        "O1CCOCCOCCOCCOCC1",
    ],
    ["bu4n", "tbab", "tbai", "tbacl", "aliquat", "18-crown-6", "18 crown 6",
     "15-crown-5", "12-crown-4", "phase transfer", "polyethylene glycol", "peg-"],
    "env"),

    ("has_protecting_reagent", [
        "Cl[Si](C)(C)C(C)(C)C",
        "Cl[Si](C)(C)C",
        "CC(C)(C)OC(=O)OC(=O)OC(C)(C)C",
        "O=C(Cl)OCc1ccccc1",
        "CC(=O)OC(C)=O",
        "BrCc1ccccc1",
        "ClCc1ccccc1",
        "C1CCOC1",
    ],
    ["tbscl", "tbdmscl", "tmscl", "boc2o", "(boc)2o", "cbz-cl", "cbzcl",
     "ac2o", "acetic anhydride", "benzyl bromide", "bn-br", "thp", "dhp",
     "momcl", "mom-cl", "fmoc-cl", "fmoc-osu", "tipscl", "trityl chloride"],
    "both"),

    ("has_radical_init",  [
        "N#CC(C)(C)/N=N/C(C)(C)C#N",
        "[SnH]([#6])([#6])[#6]",
    ],
    ["aibn", "azobisisobutyronitrile", "peroxide", "bz2o2", "(bz2o)2",
     "bu3snh", "tributyltin hydride", "(tms)3sih",
     "photo", "uv", "h\\u03bd", "hv ", "photochem", "irradiat",
     "microwave", "mw irradi", "\\u03bcw", "irradiation"],
    "env"),
]

A3_STRUCTURAL_DEFS = [
    ("has_NH",            ["[NX3;H2,H1;!$(NC=O);!$(N=*);!$([N+])]"], [], "reactant"),
    ("has_OH",            ["[OX2H1;!$(O=*);!$(OC=O)]"], [], "reactant"),
    ("has_SH",            ["[SX2H1]"], [], "reactant"),
    ("has_alkyne",        ["[CX2]#[CX2]"], [], "reactant"),
    ("has_alkene",        ["[CX3]=[CX3]"], [], "reactant"),

    ("has_aldehyde",      ["[CX3H1](=O)[#6]"], [], "reactant"),
    ("has_ketone",        ["[#6][CX3;H0](=[OX1])[#6;!$(C=[O,N,S])]"], [], "reactant"),
    ("has_ester_amide",   ["[CX3](=O)[OX2][CX4,c]",
                          "[CX3](=O)[NX3]",
                          "[CX3](=O)[SX2]"], [], "reactant"),
    ("has_anhydride_acyl_X", ["[CX3](=O)[F,Cl,Br,I]",
                              "[CX3](=O)O[CX3](=O)"], [], "reactant"),
    ("has_aryl_X",        ["c[Cl,Br,I]",
                          "cOS(=O)(=O)C(F)(F)F"], [], "reactant"),
    ("has_carboxylic_acid", ["[CX3](=O)[OX2H1]"], [], "reactant"),

]

A4_DEFS = [
    ("has_epoxide",                ["[CX4;r3]1[OX2;r3][CX4;r3]1"], [], "reactant"),
    ("has_aziridine_oxetane",      ["[CX4;r3]1[NX3;r3][CX4;r3]1",
                                    "[CX4;r4]1[OX2;r4][CX4;r4][CX4;r4]1"], [], "reactant"),
    ("has_strained_carbocycle",    ["[CX4;r3]1[CX4;r3][CX4;r3]1",
                                    "[CX4;r4]1[CX4;r4][CX4;r4][CX4;r4]1"], [], "reactant"),
    ("has_beta_lactam_lactone",    ["O=C1[NX3]CC1",
                                    "O=C1OCC1"], [], "reactant"),
    ("has_allyl_lg",               ["[Cl,Br,I][CX4][CX3]=[CX3]",
                                    "[$(O[CX3](=O)),$(OS(=O)(=O))][CX4][CX3]=[CX3]"], [], "reactant"),
    ("has_benzyl_lg",              ["[Cl,Br,I][CX4]c",
                                    "[$(O[CX3](=O)),$(OS(=O)(=O))][CX4]c"], [], "reactant"),
    ("has_1_5_diene",              ["[CX3]=[CX3][CX4][CX4][CX3]=[CX3]",
                                    "[CX3]=[CX3][O,N][CX3]=[CX3]"], [], "reactant"),

]

SIGNAL_ORDER = [d[0] for d in A1_DEFS]             + [d[0] for d in A2_DEFS]             + [d[0] for d in A3_STRUCTURAL_DEFS] + ["has_aldehyde_only"]             + ["ab_strong_acid", "ab_weak_acid", "ab_strong_base", "ab_weak_base"]             + [d[0] for d in A4_DEFS] + ["has_diene_dienophile", "has_ortho_aryl_lg", "has_spiro_center"]

assert len(SIGNAL_ORDER) == 43, f"SIGNAL_ORDER length = {len(SIGNAL_ORDER)}, expected 43"

SIGNAL_INDEX = {name: i for i, name in enumerate(SIGNAL_ORDER)}

_compiled_cache: dict[str, Optional[Chem.Mol]] = {}

def _compile(s: str) -> Optional[Chem.Mol]:
    if s not in _compiled_cache:
        _compiled_cache[s] = Chem.MolFromSmarts(s)
    return _compiled_cache[s]

def _has_any_match(mols: Sequence[Chem.Mol], smarts_list: Sequence[str]) -> bool:
    """对 mols 列表用 smarts_list 任一 pattern 找匹配, 返回是否有 ≥1 命中."""
    for s in smarts_list:
        pat = _compile(s)
        if pat is None:
            continue
        for m in mols:
            if m is None:
                continue
            try:
                if m.HasSubstructMatch(pat):
                    return True
            except Exception:
                continue
    return False

def _keyword_match(text: str, keywords: Sequence[str]) -> bool:
    """case-insensitive substring match."""
    if not text or not keywords:
        return False
    low = text.lower()
    for kw in keywords:
        if kw.lower() in low:
            return True
    return False

def _detect_smarts_signal(sig_def: tuple, r_mol: Optional[Chem.Mol],
                           env_mols: Sequence[Chem.Mol], text_blob: str) -> bool:
    """通用 SMARTS + 关键词 detector."""
    name, smarts_list, keywords, search_in = sig_def
    targets: list[Chem.Mol] = []
    if search_in in ("reactant", "both") and r_mol is not None:
        targets.append(r_mol)
    if search_in in ("env", "both"):
        targets.extend(env_mols)
    if smarts_list and _has_any_match(targets, smarts_list):
        return True
    if keywords and _keyword_match(text_blob, keywords):
        return True
    return False

def _detect_aldehyde_only(r_mol: Optional[Chem.Mol]) -> bool:
    if r_mol is None:
        return False
    ald = _compile("[CX3H1](=O)[#6]")
    ket = _compile("[CX3](=O)([#6])[#6]")
    if ald is None or ket is None:
        return False
    return r_mol.HasSubstructMatch(ald) and not r_mol.HasSubstructMatch(ket)

def _detect_diene_dienophile(r_mol: Optional[Chem.Mol]) -> bool:
    """同时含 1,3-diene 和 dienophile (烯/炔/醌 邻接 EWG)."""
    if r_mol is None:
        return False
    diene = _compile("[CX3]=[CX3][CX3]=[CX3]")
    deno1 = _compile("[CX3]=[CX3][C,N,O](=O)")
    deno2 = _compile("[CX2]#[CX2][C,N,O](=O)")
    if diene is None:
        return False
    if not r_mol.HasSubstructMatch(diene):
        return False
    if deno1 is not None and r_mol.HasSubstructMatch(deno1):
        return True
    if deno2 is not None and r_mol.HasSubstructMatch(deno2):
        return True
    return False

def _detect_ortho_aryl_lg(r_mol: Optional[Chem.Mol]) -> bool:
    """1,2-取代芳环 + 一端 leaving group, 一端 nucleophile."""
    if r_mol is None:
        return False
    pats = [
        "c1([Cl,Br,I])ccccc1[NX3;H2,H1]",
        "c1([Cl,Br,I])ccccc1[OX2H1]",
        "c1([Cl,Br,I])ccccc1[SX2H1]",
    ]
    for p in pats:
        pat = _compile(p)
        if pat is not None and r_mol.HasSubstructMatch(pat):
            return True
    return False

def _detect_spiro_center(r_mol: Optional[Chem.Mol]) -> bool:
    """sp3 C 同时属于两 ring 且 ring 仅共享该原子."""
    if r_mol is None:
        return False
    try:
        ring_info = r_mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
    except Exception:
        return False
    n = len(atom_rings)
    for i in range(n):
        si = set(atom_rings[i])
        for j in range(i + 1, n):
            sj = set(atom_rings[j])
            shared = si & sj
            if len(shared) == 1:
                idx = next(iter(shared))
                a = r_mol.GetAtomWithIdx(idx)
                if a.GetSymbol() == "C" and a.GetHybridization() == Chem.HybridizationType.SP3:
                    return True
    return False

def extract_role_signals(
    smiles_r: Optional[str],
    env_smiles_list: Optional[Sequence[str]],
    proc_text: Optional[str],
    work_text: Optional[str],
    ab_flags: Optional[Union[Sequence[int], torch.Tensor]],
) -> torch.Tensor:
    """提取单反应 43 维 SMARTS 信号. 顺序见 SIGNAL_ORDER. v25.1 拆 has_C_eq_O→4."""
    feat = torch.zeros(43, dtype=torch.float32)

    r_mol = None
    if smiles_r:
        try:
            r_mol = Chem.MolFromSmiles(smiles_r)
        except Exception:
            r_mol = None

    env_mols: list[Chem.Mol] = []
    if env_smiles_list:
        for s in env_smiles_list:
            if not s:
                continue
            try:
                m = Chem.MolFromSmiles(s)
                if m is not None:
                    env_mols.append(m)
            except Exception:
                continue

    text_blob = " ".join([t for t in (proc_text, work_text) if t])

    pos = 0
    for sig_def in A1_DEFS:
        if _detect_smarts_signal(sig_def, r_mol, env_mols, text_blob):
            feat[pos] = 1.0
        pos += 1
    for sig_def in A2_DEFS:
        if _detect_smarts_signal(sig_def, r_mol, env_mols, text_blob):
            feat[pos] = 1.0
        pos += 1
    for sig_def in A3_STRUCTURAL_DEFS:
        if _detect_smarts_signal(sig_def, r_mol, env_mols, text_blob):
            feat[pos] = 1.0
        pos += 1

    if _detect_aldehyde_only(r_mol):
        feat[pos] = 1.0
    pos += 1

    if ab_flags is not None:
        if isinstance(ab_flags, torch.Tensor):
            ab = ab_flags.flatten().tolist()
        else:
            ab = list(ab_flags)
        for k in range(min(4, len(ab))):
            feat[pos + k] = float(ab[k])
    pos += 4

    for sig_def in A4_DEFS:
        if _detect_smarts_signal(sig_def, r_mol, env_mols, text_blob):
            feat[pos] = 1.0
        pos += 1

    if _detect_diene_dienophile(r_mol):
        feat[pos] = 1.0
    pos += 1
    if _detect_ortho_aryl_lg(r_mol):
        feat[pos] = 1.0
    pos += 1
    if _detect_spiro_center(r_mol):
        feat[pos] = 1.0
    pos += 1

    assert pos == 43, f"position counter = {pos}, expected 43"
    return feat

def extract_role_signals_batch(
    batch_smiles_r: Sequence[Optional[str]],
    batch_env_smiles_list: Sequence[Optional[Sequence[str]]],
    batch_proc_text: Sequence[Optional[str]],
    batch_work_text: Sequence[Optional[str]],
    batch_ab_flags: Optional[Union[Sequence, torch.Tensor]] = None,
) -> torch.Tensor:
    """批量提取 [B, 43]."""
    B = len(batch_smiles_r)
    out = torch.zeros(B, 43, dtype=torch.float32)
    for b in range(B):
        ab = None
        if batch_ab_flags is not None:
            ab = batch_ab_flags[b] if not isinstance(batch_ab_flags, torch.Tensor) else batch_ab_flags[b]
        out[b] = extract_role_signals(
            batch_smiles_r[b],
            batch_env_smiles_list[b] if b < len(batch_env_smiles_list) else None,
            batch_proc_text[b] if b < len(batch_proc_text) else None,
            batch_work_text[b] if b < len(batch_work_text) else None,
            ab,
        )
    return out

def signal_dim_index(name: str) -> int:
    """信号名 → 维度 ID (0-39)."""
    return SIGNAL_INDEX[name]

def signal_names() -> List[str]:
    """返回 40 维信号顺序."""
    return list(SIGNAL_ORDER)

if __name__ == "__main__":

    print(f"40 signals: {SIGNAL_ORDER}")
    feat = extract_role_signals(
        smiles_r="OC(=O)c1ccccc1.NC(C)(C)C",
        env_smiles_list=["CCN=C=NCCCN(C)C", "ClCCl"],
        proc_text="stirred at room temp",
        work_text="extracted with EtOAc",
        ab_flags=[0, 0, 0, 0],
    )
    print(f"feat sum (酰胺缩合 with EDC): {feat.sum().item()}")
    print(f"  triggered: {[SIGNAL_ORDER[i] for i, v in enumerate(feat) if v > 0.5]}")
