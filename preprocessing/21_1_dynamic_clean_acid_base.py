import pandas as pd
import numpy as np
import os
import torch
import csv
import gc
from rdkit import Chem
from rxnmapper import RXNMapper
from tqdm import tqdm
import warnings
from rdkit import RDLogger
import pandas as pd
from rdkit import Chem
import gc

MAX_TOKEN_LEN = 600

UNMAPPED_PRODUCT_THRESHOLD = 0.1

_raw_solvents = [
    "CCCCCC",
    "CCCCC",
    "C1CCCCC1",
    "Cc1ccccc1",
    "CC(=O)OCC",
    "CC(=O)O",
    "CO",
    "CCO",
    "ClCCl",
    "C1CCOC1",
    "CN(C)C=O",
    "CS(=O)C",
    "CC#N",
    "ClC(Cl)Cl",
]
COMMON_SOLVENTS = set()
for s in _raw_solvents:
    m = Chem.MolFromSmiles(s)
    if m:
        COMMON_SOLVENTS.add(Chem.MolToSmiles(m))

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT_CSV = "./ord_data/ord_processed_samples_refined.csv"
OUTPUT_CSV = "./ord_data/ord_processed_samples_refined_cleaned_dataset.csv"

BATCH_SIZE = 64
SAVE_INTERVAL = 1000
MAX_TOKEN_LEN = 512
MAX_MOL_WEIGHT = 1500.0
STATS_PRINT_INTERVAL = 5000

class GlobalStats:
    def __init__(self):
        self.stats = {
            "total_input": 0,
            "success": 0,
            "filtered_empty": 0,
            "filtered_too_long": 0,
            "filtered_no_products_left": 0,
            "mapper_failed": 0,
            "filtered_no_map": 0,
            "filtered_high_unmapped": 0,
            "warning_partial_unmapped": 0,
            "water_added": 0,
            "hydrolysis_types": {},
        }

    def update(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1

    def record_hydrolysis(self, reason):
        self.stats["water_added"] += 1
        self.stats["hydrolysis_types"][reason] = (
            self.stats["hydrolysis_types"].get(reason, 0) + 1
        )

    def print_report(self, final=False):
        s = self.stats
        total = max(s["total_input"], 1)
        title = "🏁 最终统计报告" if final else "📊 实时统计报告"

        print(f"\n{'='*20} {title} {'='*20}")
        print(f"📥 输入总数: {s['total_input']}")
        print(f"✅ 成功保存: {s['success']} ({s['success']/total:.1%})")
        print(f"⚠️  部分未映射: {s['warning_partial_unmapped']} (已保留)")
        print(f"🗑️ 被过滤掉的数据:")
        print(
            f"   ├─ 产物大量未映射(>20%): {s['filtered_high_unmapped']} (关键原料缺失)"
        )
        print(f"   ├─ 产物即原料: {s['filtered_no_products_left']}")
        print(f"   ├─ 文本过长: {s['filtered_too_long']}")
        print(f"   ├─ Mapper失败: {s['mapper_failed']}")
        print(f"   └─ 其他错误: {s['filtered_empty'] + s['filtered_no_map']}")
        print(f"{'='*55}\n")

ACID_BASE_PATTERNS = {
    "strong_acid": [
        "[OH]S(=O)(=O)",
        "[Cl,Br,I;H1]",
        "[Cl,Br,I]-[H]",
        "C(=O)(C(F)(F)F)[OH]",
        "[O-][N+](=O)[O-]",
        "[Al,B,Ti,Sn,Fe,Zn][F,Cl,Br,I]",
    ],
    "weak_acid": [
        "C(=O)[OH]",
        "[NH4+]",
        "[nH]",
        "c[OH]",
        "P(=O)[OH]",
        "[#6][SH]",
    ],
    "strong_base": [
        "[Li,Na,K,Cs,Rb,Mg,Ca][H]",
        "[Li,Na,K][C]",
        "[Li,Na,K][N]",
        "[Li,Na,K][O-]",
        "[Li,Na,K][OH]",
    ],
    "weak_base": [
        "[NX3;H0,H1,H2;!$(NC=O)]",
        "c1ncccc1",
        "C(=O)([O-])[O-]",
        "C(=O)([O-])O",
        "[F-]",
    ],
}

HYDROLYSIS_PATTERNS = [
    {
        "name": "ester_hydrolysis",
        "reactant_smarts": "[CX3](=O)[OX2H0]",
        "product_smarts": "[CX3](=O)[OH]",
    },
    {
        "name": "amide_hydrolysis",
        "reactant_smarts": "[CX3](=O)[NX3]",
        "product_smarts": "[CX3](=O)[OH]",
    },
    {
        "name": "nitrile_hydrolysis",
        "reactant_smarts": "[NX1]#[CX2]",
        "product_smarts": "[CX3](=O)[OH,N]",
    },
    {
        "name": "acyl_halide_hydrolysis",
        "reactant_smarts": "[CX3](=O)[F,Cl,Br,I]",
        "product_smarts": "[CX3](=O)[OH]",
    },
]

COMPILED_AB_PATTERNS = {}
for k, v in ACID_BASE_PATTERNS.items():
    valid = []
    for s in v:
        p = Chem.MolFromSmarts(s)
        if p:
            valid.append(p)
    COMPILED_AB_PATTERNS[k] = valid

for p in HYDROLYSIS_PATTERNS:
    p["r_pat"] = Chem.MolFromSmarts(p["reactant_smarts"])
    p["p_pat"] = Chem.MolFromSmarts(p["product_smarts"])

def detect_acid_base(smiles_list):
    """检测环境分子列表中的酸碱性"""
    flags = {f"is_{k}": False for k in ACID_BASE_PATTERNS.keys()}
    if not smiles_list:
        return flags

    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue

        for cat, patterns in COMPILED_AB_PATTERNS.items():
            if flags[f"is_{cat}"]:
                continue
            for pat in patterns:
                if mol.HasSubstructMatch(pat):
                    flags[f"is_{cat}"] = True
                    break
    return flags

def count_oxygen(smiles):
    if not smiles:
        return 0
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)

def check_hydrolysis_need_water(r_mols, p_mols):
    """
    智能补水判断逻辑
    返回: (bool 是否需要补水, str 原因)
    """

    for s in r_mols:
        if s == "O" or s == "[OH2]":
            return False, "water_exists"

    r_o = sum(count_oxygen(s) for s in r_mols)
    p_o = sum(count_oxygen(s) for s in p_mols)
    if p_o > r_o:
        return True, "oxygen_imbalance"

    try:

        r_big = Chem.MolFromSmiles(".".join(r_mols))
        p_big = Chem.MolFromSmiles(".".join(p_mols))
    except:
        return False, ""

    if not r_big or not p_big:
        return False, ""

    for pat in HYDROLYSIS_PATTERNS:

        r_matches = len(r_big.GetSubstructMatches(pat["r_pat"]))
        p_matches = len(p_big.GetSubstructMatches(pat["p_pat"]))

        if p_matches > 0 and r_matches > 0:
            return True, pat["name"]

    return False, ""

class GPUInferenceWrapper(torch.nn.Module):
    """
    劫持 RXNMapper 内部模型，强制其在 GPU 上运行并返回兼容的 Output 对象
    """

    def __init__(self, original_model, device):
        super().__init__()
        self.original_model = original_model.to(device)
        self.device = device
        self.config = original_model.config

    def forward(self, **kwargs):

        gpu_kwargs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
        }

        with torch.no_grad():
            outputs = self.original_model(**gpu_kwargs)

        if hasattr(outputs, "attentions"):
            cpu_attentions = tuple(att.cpu() for att in outputs.attentions)
        elif isinstance(outputs, (tuple, list)):
            cpu_attentions = tuple(att.cpu() for att in outputs[-1])
        else:
            cpu_attentions = ()

        class HybridOutput:
            def __init__(self, atts):
                self.attentions = atts

            def __getitem__(self, item):
                if item == "attentions" or item == -1 or item == 2:
                    return self.attentions
                return None

            def __len__(self):
                return 3

            def __iter__(self):
                yield from [None, None, self.attentions]

        return HybridOutput(cpu_attentions)

def enable_gpu_acceleration(rxn_mapper, device):
    print(f"🔧 正在挂载 GPU 加速引擎 (Device: {device})...")
    rxn_mapper.model = GPUInferenceWrapper(rxn_mapper.model, device)
    print("✅ GPU 引擎挂载成功！")

def get_canonical_smiles(smi):
    if not smi:
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        pass
    return None

def get_all_precursors(row):
    """合并四列前体物质"""
    candidates = set()
    cols = ["Reactants", "Solvents", "Catalysts", "Reagents"]
    for col in cols:
        val = row.get(col)
        if pd.isna(val) or str(val) in ["nan", "None"]:
            continue
        mols = [s.strip() for s in str(val).split(";") if s.strip()]
        candidates.update(mols)
    return list(candidates)

def is_pure_alkane(mol):
    """检测是否为纯烷烃"""
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (1, 6):
            return False
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            return False
    return True

def has_new_heteroatoms(r_mol, p_mol_str):
    """检测产物是否凭空出现了反应物没有的杂原子"""
    p_mol = Chem.MolFromSmiles(p_mol_str)
    if not p_mol:
        return False
    r_atoms = set(a.GetAtomicNum() for a in r_mol.GetAtoms())
    p_atoms = set(a.GetAtomicNum() for a in p_mol.GetAtoms())
    for atom_num in p_atoms:
        if atom_num > 6 and atom_num not in r_atoms:
            return True
    return False

def clean_batch_robust(df_batch, rxn_mapper, stats_tracker):
    rxn_inputs = []
    water_added_flags = []
    hydrolysis_types = []
    temp_rows = []

    for _, row in df_batch.iterrows():
        stats_tracker.update("total_input")

        p_str = str(row["Products"])
        if not p_str.strip() or p_str == "nan":
            stats_tracker.update("filtered_empty")
            continue
        raw_p_mols = [s.strip() for s in p_str.split(";") if s.strip()]

        all_candidates = get_all_precursors(row)
        if not all_candidates:
            stats_tracker.update("filtered_empty")
            continue

        precursor_fingerprints = set()
        for smi in all_candidates:
            can = get_canonical_smiles(smi)
            if can:
                precursor_fingerprints.add(can)

        valid_p_mols = []
        for p_smi in raw_p_mols:
            p_can = get_canonical_smiles(p_smi)
            if p_can is None or p_can not in precursor_fingerprints:
                valid_p_mols.append(p_smi)

        if not valid_p_mols:
            stats_tracker.update("filtered_no_products_left")
            continue

        need_water, reason = check_hydrolysis_need_water(all_candidates, valid_p_mols)
        if need_water:
            if "O" not in all_candidates and "[OH2]" not in all_candidates:
                all_candidates.append("O")
                stats_tracker.record_hydrolysis(reason)

        rxn_str = f"{'.'.join(all_candidates)}>>{'.'.join(valid_p_mols)}"
        if len(rxn_str) > MAX_TOKEN_LEN:
            stats_tracker.update("filtered_too_long")
            continue

        rxn_inputs.append(rxn_str)
        water_added_flags.append(need_water)
        hydrolysis_types.append(reason)
        temp_rows.append(row)

    if not rxn_inputs:
        return []

    results = []
    try:
        results = rxn_mapper.get_attention_guided_atom_maps(rxn_inputs)
    except:

        for r in rxn_inputs:
            try:
                results.append(rxn_mapper.get_attention_guided_atom_maps([r])[0])
            except:
                results.append(None)

    processed_rows = []
    for i, res_obj in enumerate(results):
        if (
            not res_obj
            or "mapped_rxn" not in res_obj
            or ">>" not in res_obj["mapped_rxn"]
        ):
            stats_tracker.update("mapper_failed")
            continue

        mapped_rxn = res_obj["mapped_rxn"]
        mapped_r_str, mapped_p_str = mapped_rxn.split(">>")
        original_row = temp_rows[i]

        p_mol_obj = Chem.MolFromSmiles(mapped_p_str)
        if not p_mol_obj:
            stats_tracker.update("mapper_failed")
            continue

        product_map_ids = set()
        p_heavy_total = 0
        p_heavy_unmapped = 0

        for atom in p_mol_obj.GetAtoms():
            if atom.GetAtomicNum() > 1:
                p_heavy_total += 1
                mid = atom.GetAtomMapNum()
                if mid > 0:
                    product_map_ids.add(mid)
                else:
                    p_heavy_unmapped += 1

        if p_heavy_total == 0:
            stats_tracker.update("filtered_empty")
            continue

        unmapped_ratio = p_heavy_unmapped / p_heavy_total

        if unmapped_ratio > UNMAPPED_PRODUCT_THRESHOLD:

            stats_tracker.update("filtered_high_unmapped")
            continue
        elif p_heavy_unmapped > 0:

            stats_tracker.update("warning_partial_unmapped")

        if not product_map_ids:
            stats_tracker.update("filtered_no_map")
            continue

        true_reactants = []
        environment_molecules = []
        r_fragments = mapped_r_str.split(".")

        for frag in r_fragments:
            frag_mol = Chem.MolFromSmiles(frag)
            if not frag_mol:
                continue

            n_heavy_total = 0
            n_heavy_mapped = 0
            mapped_elements = set()

            for atom in frag_mol.GetAtoms():
                if atom.GetAtomicNum() > 1:
                    n_heavy_total += 1
                    if atom.GetAtomMapNum() in product_map_ids:
                        n_heavy_mapped += 1
                        mapped_elements.add(atom.GetAtomicNum())

            ratio = n_heavy_mapped / max(n_heavy_total, 1)

            is_contributor = False

            if n_heavy_mapped > 0:

                if n_heavy_total <= 4:
                    is_contributor = True

                elif n_heavy_mapped >= 2 or ratio > 0.3:
                    is_contributor = True

                else:
                    has_heteroatom_mapped = any(z != 6 for z in mapped_elements)
                    if has_heteroatom_mapped:
                        is_contributor = True

            if is_contributor:
                frag_can = Chem.MolToSmiles(frag_mol)

                if is_pure_alkane(frag_mol) and has_new_heteroatoms(
                    frag_mol, mapped_p_str
                ):
                    is_contributor = False

                elif frag_can in COMMON_SOLVENTS:

                    if ratio > 0.15:
                        is_contributor = True
                    else:
                        is_contributor = False

            if is_contributor:
                true_reactants.append(frag)
            else:

                [a.SetAtomMapNum(0) for a in frag_mol.GetAtoms()]
                environment_molecules.append(Chem.MolToSmiles(frag_mol))

        environment_molecules = list(set(environment_molecules))
        ab_flags = detect_acid_base(environment_molecules)

        try:
            y = float(original_row["Yield"])
            y_int = -1 if y < 0 else int(round(y))
        except:
            y_int = -1

        new_row = {
            "Reaction ID": original_row.get("Reaction ID", ""),
            "Yield": y_int,
            "Mapped_Reactants": " . ".join(true_reactants),
            "Mapped_Products": mapped_p_str,
            "Environment_Molecules": " ; ".join(environment_molecules),
            "Temperature": original_row.get("Temperature(C)", ""),
            "Time": original_row.get("Time(h)", ""),
            "Is_Strong_Acid": ab_flags["is_strong_acid"],
            "Is_Weak_Acid": ab_flags["is_weak_acid"],
            "Is_Strong_Base": ab_flags["is_strong_base"],
            "Is_Weak_Base": ab_flags["is_weak_base"],
            "Water_Added": water_added_flags[i],
            "Hydrolysis_Type": hydrolysis_types[i],
            "Unmapped_Product_Ratio": round(unmapped_ratio, 3),
            "Procedure_Text": original_row.get("Procedure_Text", ""),
            "Workup_Text": original_row.get("Workup_Text", ""),
        }
        processed_rows.append(new_row)
        stats_tracker.update("success")

    return processed_rows

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 运行设备: {device}")

    stats = GlobalStats()

    print("⏳ 加载 RXNMapper 模型...")
    rxn_mapper = RXNMapper()
    if device.type == "cuda":
        enable_gpu_acceleration(rxn_mapper, device)

    print(f"📂 读取数据: {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print("❌ 输入文件不存在")
        return

    total_rows = sum(1 for _ in open(INPUT_CSV)) - 1

    start_idx = 0
    file_mode = "w"
    header = True
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, "r") as f:
            processed_count = sum(1 for _ in f) - 1
        if processed_count > 0:
            print(f"🔄 断点续传: 跳过前 {processed_count} 条")
            start_idx = processed_count
            file_mode = "a"
            header = False

    chunk_iterator = pd.read_csv(
        INPUT_CSV, chunksize=SAVE_INTERVAL, skiprows=range(1, start_idx + 1)
    )

    pbar = tqdm(total=total_rows, initial=start_idx, unit="rxn")
    buffer = []

    processed_since_last_print = 0

    for chunk_df in chunk_iterator:
        for i in range(0, len(chunk_df), BATCH_SIZE):
            batch_df = chunk_df.iloc[i : i + BATCH_SIZE]

            cleaned_batch = clean_batch_robust(batch_df, rxn_mapper, stats)

            buffer.extend(cleaned_batch)
            pbar.update(len(batch_df))
            processed_since_last_print += len(batch_df)

            if stats.stats["total_input"] > 0:
                acc = stats.stats["success"] / stats.stats["total_input"]
                pbar.set_postfix(acc=f"{acc:.1%}", water=stats.stats["water_added"])

        if processed_since_last_print >= STATS_PRINT_INTERVAL:
            stats.print_report()
            processed_since_last_print = 0

        if buffer:
            out_df = pd.DataFrame(buffer)
            cols_order = [
                "Reaction ID",
                "Yield",
                "Mapped_Reactants",
                "Mapped_Products",
                "Environment_Molecules",
                "Temperature",
                "Time",
                "Is_Strong_Acid",
                "Is_Weak_Acid",
                "Is_Strong_Base",
                "Is_Weak_Base",
                "Water_Added",
                "Hydrolysis_Type",
                "Unmapped_Product_Ratio",
                "Procedure_Text",
                "Workup_Text",
            ]
            cols_order = [c for c in cols_order if c in out_df.columns]
            out_df = out_df[cols_order]
            out_df.to_csv(OUTPUT_CSV, mode=file_mode, header=header, index=False)
            header = False
            file_mode = "a"
            buffer = []
            gc.collect()

    pbar.close()
    stats.print_report(final=True)
    print(f"\n✅ 全部完成！结果已保存至 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
