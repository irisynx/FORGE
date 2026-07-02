import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except RuntimeError:
    pass
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, LayerNorm, GlobalAttention
from torch_geometric.utils import to_dense_batch
from transformers import AutoModel, AutoTokenizer
from rdkit import RDLogger
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import math
import copy
import time
import warnings
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from forge_rc_modules import (
    RCSetPredHead,
    compute_set_prediction_loss,
    predict_rc_atoms_from_set,
)

from forge_edit_modules import DETREditHead

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.utils.checkpoint"
)

RDLogger.DisableLog("rdApp.*")

torch.set_float32_matmul_precision("high")

try:
    from forge_role_signals import extract_role_signals as _extract_role_signals
except Exception as _e:
    _extract_role_signals = None
    print(f"[v25] 警告: tools.reagent_role_signals 不可用 ({_e}), use_role_signals 将自动失效")

_ROLE_SIGNAL_CACHE: dict = {}
_ROLE_SIGNAL_CACHE_MAX = 600_000

def setup_ddp():
    """初始化分布式训练。返回 (rank, world_size, is_ddp)"""
    if "RANK" in os.environ:

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, True
    else:

        gpu_id = CONFIG.get("single_gpu", 0)
        torch.cuda.set_device(gpu_id)
        return gpu_id, 1, False

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

CONFIG = {
    "device": None,
    "single_gpu": 0,
    "data_dir": "./ord_data/ord_train",
    "val_dir":  "./ord_data/ord_heldout",
    "elec_ckpt_path": "./checkpoints/model_epoch_200.pth",
    "save_dir": "./results_editgnn_v33_J",
    "data_ratio": 1.0,

    "resume_from_epoch": 21,
    "warmstart_from": "./results_editgnn_v29_2/model_best.pth",
    "warmstart_strict": False,
    "new_module_lr_mult": 1.5,
    "loader_batch_size": 4,
    "accum_steps": 3,
    "epochs": 30,
    "num_workers": 6,
    "prefetch_factor": 4,
    "use_compile": False,

    "ddp_find_unused_parameters": True,
    "ddp_gradient_as_bucket_view": True,
    "ddp_broadcast_buffers": False,

    "edge_transformer_use_checkpoint": True,

    "cosine_T_0":   12,
    "cosine_T_mult": 2,

    "lr_bert": 2e-5,
    "lr_rc": 2e-4,
    "lr_edit": 4e-5,
    "lr_edit_inter_new": 1.5e-4,
    "lr_encoder_top": 3e-5,
    "lr_cross_frag": 3e-4,
    "unfreeze_encoder_top_n": 2,

    "hidden_dim": 512,
    "gnn_layers": 6,
    "gnn_heads": 8,
    "delta_classes": 7,
    "bert_model": "./scibert_local",

    "w_rc": 1.5,
    "w_edit": 1.0,

    "rc_pos_weight": 5.0,

    "phase2_start_epoch": 1,

    "phase2_warmup_epochs": 1,
    "phase2_lr_factor": 0.1,
    "phase2_rc_freeze_epochs": 3,
    "phase2_rc_pos_weight": 6.0,

    "ss_start_prob": 0.0,
    "ss_end_prob": 0.5,
    "ss_warmup_epochs": 25,
    "ss_rc_threshold": 0.5,
    "rc_refine_enabled": False,
    "soft_rc_threshold": 0.3,

    "focal_gamma": 1.3,

    "focal_alpha": [1.0, 4.0, 4.0, 3.5, 1.5, 5.0, 5.0],
    "focal_alpha_warmup_epochs": 5,

    "use_motif_hard_neg":           True,
    "motif_hard_neg_rules": [
        (0, 1, 6, 7, 2.5),
        (0, 2, 6, 8, 2.0),
        (0, 1, None, None, 1.5),
        (0, 2, None, None, 1.5),
    ],
    "motif_hard_neg_warmup_epochs": 15,

    "use_cc_pair_constraint":       True,
    "w_cc_pair_constraint":         0.05,
    "cc_pair_warmup_epochs":        20,

    "use_large_cand_dropout":       True,
    "large_cand_dropout_threshold": 100,
    "large_cand_dropout_warmup_eps": 5,
    "large_cand_dropout_max":       0.3,

    "edit_nz_acc_collapse_threshold": 0.86,

    "rc_min_top_k": 8,
    "rc_all_pairs": True,
    "max_rc_atoms": 30,
    "max_cand_edges_per_mol": 200,
    "gt_drop_prob": 0.0,

    "cross_frag_layers": 3,
    "cross_frag_heads": 8,

    "rc_hard_neg_ratio": 0.3,
    "rc_hard_neg_ratio_start": 0.5,
    "rc_hard_neg_ratio_end": 0.15,
    "rc_hard_neg_weight": 5.0,
    "rc_hard_neg_weight_end": 7.0,
    "rc_hard_neg_curriculum_epochs": 30,
    "rc_neg_weight_warmup_epochs": 10,

    "rc_focal_gamma": 1.5,
    "rc_focal_gamma_neg": 1.0,
    "rc_focal_phase2_gamma": 0.5,
    "rc_element_reweight": True,
    "rc_elem_w_F_pos": 2.0,
    "rc_elem_w_halide_neg": 3.0,
    "rc_elem_w_C_arom_pos": 1.3,
    "rc_elem_w_C_alip_pos": 1.3,
    "rc_elem_w_P_neg": 5.0,
    "rc_elem_w_Br_neg": 2.5,

    "w_pair_sym": 0.15,
    "pair_sym_alpha": 1.0,
    "pair_sym_margin": 0.7,
    "pair_sym_warmup_epochs": 10,

    "w_pair_contrastive": 0.05,
    "pair_contrastive_warmup_epochs": 12,
    "pair_contrastive_temp": 0.1,

    "w_bond_rc": 0.15,
    "bond_rc_pos_weight": 10.0,

    "rc_hot_bin_reweight": True,

    "rc_pair_refine_top_k": 6,

    "aug_edge_drop_ratio": 0.03,
    "aug_feat_mask_ratio": 0.05,
    "aug_subgraph_mask_ratio": 0.03,
    "adapter_rc_dropout": 0.15,
    "adapter_edit_dropout": 0.1,
    "rc_head_dropout": 0.2,
    "rc_label_smoothing": 0.05,

    "nan_patience": 3,
    "nan_epoch_patience": 2,
    "loss_spike_ratio": 5.0,
    "edit_loss_clamp": 8.0,
    "edit_collapse_threshold": 0.3,

    "frag_full_attn_layers": 2,
    "phase1_calibrate_temperature": True,

    "max_frag_atoms": 400,

    "use_global_anchor":        True,
    "n_global_anchors":         8,
    "n_global_anchor_layers":   3,
    "global_anchor_heads":      8,
    "global_anchor_dropout":    0.1,

    "n_edge_transformer_layers": 4,

    "use_pair_sym_head":        True,
    "pair_head_hidden":         256,
    "w_pair_head":              0.0,
    "w_pair_teacher":           0.0,
    "pair_head_warmup_epochs":  5,
    "pair_teacher_neg_weight":  0.3,
    "pair_head_top_k":          15,

    "pair_teacher_asymmetric":  True,
    "pair_teacher_margin":      0.05,

    "use_topk_fp_hard_neg":     True,
    "topk_fp_per_reaction":     5,
    "w_topk_fp_hard_neg":       2.0,
    "topk_fp_warmup_epochs":    2,

    "use_pair_hard_head":        True,
    "w_pair_hard":               0.3,
    "pair_hard_pos_weight":      3.0,
    "pair_hard_top_k":           15,
    "pair_hard_warmup_epochs":   3,
    "pair_hard_hidden":          256,
    "pair_hard_dropout":         0.1,

    "use_all_atom_hit_loss":       True,
    "w_all_atom_hit":              0.2,
    "all_atom_hit_warmup_epochs":  2,

    "use_pair_cons_gate":          False,
    "pair_cons_alpha_target":      0.3,
    "pair_cons_warmup_start":      0,
    "pair_cons_warmup_end":        3,
    "pair_cons_use_relu":          True,
    "pair_cons_infer":             True,
    "lr_pair_cons":                3e-4,

    "use_valence_dh":              True,
    "w_valence_dh":                0.05,
    "valence_dh_atom_filter":      "gt_rc",
    "valence_dh_use_softmax":      True,
    "valence_dh_warmup_epochs":    1,

    "use_valence_rc_penalty":      False,
    "w_valence_rc":                0.10,
    "valence_rc_cap_saturated":    0.3,
    "valence_rc_warmup_epochs":    0,
    "use_pair_exist_constraint":   False,
    "w_pair_exist":                0.05,
    "pair_exist_margin":           0.2,
    "pair_exist_warmup_epochs":    2,

    "use_electron_conservation":   False,
    "w_electron_cons":             0.05,
    "use_bond_order_bound":        False,
    "w_bond_order_bound":          0.10,

    "use_fukui_projection":        False,
    "fukui_proj_dim":              64,
    "charge_proj_dim":              32,
    "fukui_proj_dropout":          0.1,

    "use_fmo_atom_injection":      False,
    "fmo_proj_dim":                32,

    "frag_desc_fukui_channels":    ["mean"],

    "use_fukui_rank_loss":         False,
    "w_fukui_rank":                0.05,
    "fukui_rank_margin":           0.01,
    "fukui_rank_warmup_epochs":    0,

    "warmstart_pad_init_scale":    0.1,

    "use_pfrh":                    False,
    "pfrh_eps":                    1e-5,
    "pfrh_min_frag":               3,
    "pfrh_init_T":                 1.0,
    "pfrh_init_b":                 0.0,

    "use_mol_aux_loss":            False,
    "w_mol_aux":                   0.0,
    "mol_aux_temperature_atom":    0.02,
    "mol_aux_clean_threshold":     0.95,

    "use_eaml":                    False,
    "w_eaml":                      0.10,
    "mol_aux_temperature_edit":    0.05,

    "use_rlc_film":                True,
    "rlc_dim":                     512,
    "rlc_dropout":                 0.1,

    "use_edit_rlc":                True,

    "edit_ramp_ceiling":           0.6,

    "rc_logit_clamp":              15.0,

    "rc_f1_collapse_threshold":    0.93,

    "phase2_collapse_grace_epochs": 5,

    "lr_rc_phase2_floor":          2e-6,

    "rlc_film_edit_detach":        True,

    "rc_neg_prob_threshold":       0.20,
    "rc_neg_prob_step_ratio":      3.0,

    "rc_pred_count_ratio_threshold": 1.5,

    "use_set_recall_loss":     False,
    "use_set_precision_loss":  False,
    "w_set_recall":            0.0,
    "w_set_precision":         0.0,
    "set_loss_warmup_epochs":  0,
    "set_loss_p_clamp_min":    1e-6,
    "set_loss_max_clamp":      5.0,

    "set_loss_k_weights": {
        0:    0.5,
        2:    0.6,
        3:    1.0,
        4:    2.8,
        "5+": 3.5,
    },
    "set_precision_min_K":     4,
    "set_loss_per_fragment":   False,

    "watchdog_K2_baseline":    0.924,
    "watchdog_K2_tolerance":   0.01,
    "watchdog_K2_grace_epochs": 3,

    "use_main_product_judgment": True,
    "save_best_by_main_acc":    True,

    "use_main_edit_weight":      True,
    "main_edit_loss_mult":       1.5,
    "non_main_edit_loss_mult":   0.5,
    "main_edit_warmup_epochs":   3,

    "use_K_pos_edge_weight":      False,
    "K_pos_edge_K4_mult":         1.0,
    "K_pos_edge_K5p_mult":        1.0,
    "K_pos_edge_warmup_epochs":   2,

    "use_rc_refiner":            True,
    "rc_refiner_layers":         2,
    "rc_refiner_heads":          8,
    "rc_refiner_dropout":        0.1,
    "use_pointer_rc_loss":       True,
    "w_pointer_rc":              0.20,

    "use_edit_refiner":          False,

    "pointer_rc_warmup_end":     5,
    "pointer_rc_decay_end":      8,

    "early_stop_patience":       8,

    "early_stop_metric":         "val_K_pos_product_mol_acc",
    "save_best_metric":          "K_pos_product_mol_acc",
    "use_product_level_eval":   True,

    "use_curriculum":            False,
    "curriculum_phase2a_end":    8,
    "curriculum_phase2b_end":    12,
    "curriculum_phase2b_k23_ratio": 0.30,
    "curriculum_phase2c_lr_factor": 0.5,

    "watchdog_K2_threshold":     0.93,
    "watchdog_K3_threshold":     0.88,
    "watchdog_K2_phase2c":       0.92,

    "use_margin_penalty":         True,
    "w_margin_penalty":           0.05,
    "margin_penalty_warmup_end":  5,

    "use_mutual_exclusion":       True,
    "w_mutual_exclusion":         0.05,
    "mutual_exclusion_warmup_end": 5,
    "mutual_exclusion_min_edges": 2,

    "use_edit_conditional_rc":    False,
    "w_edit_conditional_rc":      0.10,

    "use_l1_sparsity":            False,
    "w_l1_sparsity":              0.01,

    "use_role_signals":            True,
    "role_signal_dim":             43,
    "role_signal_dropout":         0.1,

    "role_signal_proj_freeze_epochs": 0,
    "role_signals_recompute":      False,

    "use_multi_hop_rc":            False,
    "multi_hop_rc_layers":         2,
    "multi_hop_rc_gate_init":      "zero",
    "multi_hop_rc_warmup_epochs":  2,

    "use_k_aware_sparsity_loss":   False,
    "w_k_aware_sparsity":          0.0,
    "k_aware_target_pad":          1,
    "k_aware_warmup_epochs":       3,

    "use_k_pred_head":             False,
    "k_pred_head_classes":         8,
    "k_pred_head_proj_hidden":     256,
    "k_pred_head_dropout":         0.1,
    "w_k_pred":                    0.0,
    "k_pred_warmup_epochs":        0,
    "k_pred_class_weights":        [1.0] * 8,
    "use_k_pred_inference_nms":    False,

    "use_rc_set_pred_head":        True,
    "rc_set_pred_K_max":           10,
    "rc_set_pred_num_layers":      3,
    "rc_set_pred_num_heads":       8,
    "rc_set_pred_ffn_dim":         2048,
    "rc_set_pred_dropout":         0.1,
    "rc_set_pred_N_max":           60,
    "rc_set_pred_tau":             0.5,
    "rc_set_pred_lambda_no":       0.5,
    "rc_set_pred_alpha_cost":      1.0,
    "w_rc_set_pred":               1.0,
    "rc_set_pred_lr":              5e-4,

    "use_rc_main_weight":          True,
    "rc_main_loss_mult":           1.1,
    "rc_non_main_loss_mult":       0.8,
    "rc_main_warmup_epochs":       3,

    "use_rc_symmetry_expansion":   False,
    "rc_sym_max_swap_per_atom":    4,
    "rc_sym_skip_high_K":          True,
    "rc_sym_skip_K_threshold":     4,
    "use_rc_sym_stats_logging":    False,

    "use_edit_temperature":        False,
    "edit_temperature":            1.0,

    "use_equivalence_class_aug":   True,
    "equiv_aug_prob":              0.5,
    "equiv_aug_prob_active":       0.0,
    "equiv_aug_warmup_epochs":     2,

    "cross_frag_attn_freeze_epochs":    2,

    "inference_hook_A_tau":        0.40,
    "inference_hook_V_mono":       True,

    "edge_feat_use_interaction_terms": True,
}

CONFIG.update({

    "save_dir":                       "./results_editgnn_v34_E1",
    "warmstart_from":                 "./results_editgnn_v33_J/model_best.pth",
    "resume_from_epoch":              0,
    "single_gpu":                     0,

    "use_detr_edit_head":             True,
    "detr_edit_n_decoder_layers":     2,
    "detr_edit_n_heads":              8,
    "detr_edit_hidden":               512,
    "detr_edit_ffn_dim":              2048,
    "detr_edit_dropout":              0.1,
    "detr_edit_init_from_v29_weight": True,

    "use_old_edit_classifier_fallback": True,
    "old_edit_classifier_loss_weight":  0.0,

    "phase0_head_only_epochs":        5,
    "phase1_start_epoch":             6,
    "phase2_start_epoch":             1,
    "current_phase_force":            2,

    "lr_bert":                        0.0,
    "lr_rc":                          0.0,
    "lr_edit":                        0.0,
    "lr_edit_inter_new":              0.0,
    "lr_encoder_top":                 0.0,
    "lr_cross_frag":                  0.0,
    "lr_pair_cons":                   0.0,
    "rc_set_pred_lr":                 0.0,
    "lr_adapter_edit_phase0":         0.0,
    "lr_adapter_edit_phase1":         5e-6,
    "lr_detr_edit_phase0":            5e-4,
    "lr_detr_edit_phase1":            1e-4,
    "lr_edit_classifier":             0.0,
    "unfreeze_encoder_top_n":         0,

    "grad_clip_norms": [1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0],

    "w_rc":                           0.0,
    "w_rc_set_pred":                  0.0,
    "w_edit":                         1.0,
    "w_edit_phase0":                  1.0,
    "w_edit_phase1":                  1.0,

    "w_pair_sym":                     0.0,
    "w_pair_contrastive":             0.0,
    "w_bond_rc":                      0.0,
    "w_cc_pair_constraint":           0.0,
    "w_topk_fp_hard_neg":             0.0,
    "w_pair_hard":                    0.0,
    "w_all_atom_hit":                 0.0,
    "w_valence_dh":                   0.0,
    "w_valence_rc":                   0.0,
    "w_pair_exist":                   0.0,
    "w_electron_cons":                0.0,
    "w_bond_order_bound":             0.0,
    "w_fukui_rank":                   0.0,
    "w_eaml":                         0.0,
    "w_pointer_rc":                   0.0,
    "w_margin_penalty":               0.0,
    "w_mutual_exclusion":             0.0,
    "w_edit_conditional_rc":          0.0,
    "w_l1_sparsity":                  0.0,

    "phase2_rc_freeze_epochs":        999,
    "cross_frag_attn_freeze_epochs":  999,
    "phase2_lr_factor":               1.0,
    "phase2_warmup_epochs":           1,

    "epochs":                         30,
    "loader_batch_size":              4,
    "accum_steps":                    4,
    "num_workers":                    6,
    "prefetch_factor":                4,

})

CONFIG.update({

    "save_dir":                       "./results_editgnn_v34_E1_uspto_v2",
    "data_dir":                       "./uspto_480k/train",
    "val_dir":                        "./uspto_480k/valid",
    "warmstart_from":                 None,
    "resume_from_epoch":              0,
    "single_gpu":                     0,
    "epochs":                         30,

    "phase2_warmup_epochs":           5,
    "edit_ramp_ceiling":              1.0,

    "lr_bert":                        2e-5,
    "lr_rc":                          2e-4,
    "lr_edit":                        4e-5,
    "lr_edit_inter_new":              1.5e-4,
    "lr_encoder_top":                 3e-5,
    "lr_cross_frag":                  3e-4,
    "lr_pair_cons":                   1e-4,
    "rc_set_pred_lr":                 2e-4,
    "lr_adapter_edit_phase0":         5e-5,
    "lr_adapter_edit_phase1":         5e-5,
    "lr_detr_edit_phase0":            5e-4,
    "lr_detr_edit_phase1":            1e-4,
    "lr_edit_classifier":             4e-5,
    "unfreeze_encoder_top_n":         2,

    "w_rc":                           1.5,
    "w_rc_set_pred":                  1.0,
    "w_edit":                         1.0,

    "phase2_rc_freeze_epochs":        0,
    "cross_frag_attn_freeze_epochs":  0,

    "detr_edit_init_from_v29_weight": False,

    "rc_f1_collapse_threshold":      0.0,
    "phase2_collapse_grace_epochs":  30,
    "rc_neg_prob_threshold":         1.0,
    "rc_neg_prob_step_ratio":        100.0,
    "rc_pred_count_ratio_threshold": 100.0,
    "watchdog_K2_baseline":          0.0,
    "watchdog_K2_tolerance":         1.0,
    "watchdog_K2_grace_epochs":      30,

    "resume_from_epoch":             0,
})

ATOM_FEATURE_DIMS = [120, 15, 15, 10, 10, 100, 10, 10]

CLASS_TO_DELTA = torch.tensor([0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0])

MAX_VALENCE = {
    6: 4,
    7: 3,
    8: 2,
    9: 1,
    15: 5,
    16: 6,
    17: 1,
    35: 1,
    53: 1,
    5: 3,
    14: 4,
    33: 5,
    34: 6,
}

os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["bert_model"])
except Exception:
    print("Warning: Could not load SciBERT tokenizer.")
    TOKENIZER = None

_TOKEN_CACHE = {}

def random_permute_graph(data: Data):
    num_nodes = data.x.size(0)
    perm = torch.randperm(num_nodes)
    data.x = data.x[perm]
    inv_perm = torch.zeros_like(perm)
    inv_perm[perm] = torch.arange(num_nodes)
    data.edge_index = inv_perm[data.edge_index]
    if hasattr(data, "y_delta") and data.y_delta is not None:
        if data.y_delta.dim() == 2:
            data.y_delta = data.y_delta[perm][:, perm]
    return data

def random_edge_dropout(data: Data, drop_ratio=0.1):
    """随机删除非关键边，迫使模型不过度依赖完整拓扑结构。
    保护涉及键级变化的边（y_delta 非零），只删除无变化的边。"""
    if drop_ratio <= 0 or data.edge_index.size(1) == 0:
        return data
    num_edges = data.edge_index.size(1)

    protected = torch.zeros(num_edges, dtype=torch.bool)
    if hasattr(data, "y_delta") and data.y_delta is not None and data.y_delta.dim() == 2:
        src, dst = data.edge_index[0], data.edge_index[1]
        protected = data.y_delta[src, dst] != 0

    drop_mask = torch.rand(num_edges) < drop_ratio
    drop_mask = drop_mask & ~protected
    keep_mask = ~drop_mask
    data.edge_index = data.edge_index[:, keep_mask]
    data.edge_attr = data.edge_attr[keep_mask]
    return data

def random_feature_mask(data: Data, mask_ratio=0.15):
    """随机遮蔽原子特征列，防止过拟合单一化学特征。
    对 x[N,8] 的每一列独立以 mask_ratio 概率置 0。"""
    if mask_ratio <= 0:
        return data
    num_nodes, num_feats = data.x.shape

    mask = torch.rand(num_nodes, num_feats) < mask_ratio
    data.x = data.x.clone()
    data.x[mask] = 0
    return data

def random_subgraph_mask(data: Data, mask_ratio=0.1):
    """随机 mask 非反应中心原子的特征，训练不完整信息下的鲁棒性。
    RC 原子（y_delta 行/列非零）不受影响。"""
    if mask_ratio <= 0:
        return data
    num_nodes = data.x.size(0)

    rc_mask = torch.zeros(num_nodes, dtype=torch.bool)
    if hasattr(data, "y_delta") and data.y_delta is not None and data.y_delta.dim() == 2:
        row_active = (data.y_delta != 0).any(dim=1)
        col_active = (data.y_delta != 0).any(dim=0)
        rc_mask = row_active | col_active

    non_rc = ~rc_mask
    non_rc_indices = non_rc.nonzero(as_tuple=True)[0]
    if len(non_rc_indices) == 0:
        return data
    num_to_mask = max(1, int(len(non_rc_indices) * mask_ratio))
    perm = torch.randperm(len(non_rc_indices))[:num_to_mask]
    mask_indices = non_rc_indices[perm]
    data.x = data.x.clone()
    data.x[mask_indices] = 0
    return data

_ATOM_MAX_VALENCE_LIST = [4] * 120
for _z, _v in [
    (1, 1),
    (5, 3),
    (6, 4),
    (7, 3),
    (8, 2),
    (9, 1),
    (14, 4),
    (15, 5),
    (16, 6),
    (17, 1),
    (35, 1),
    (53, 1),
]:
    _ATOM_MAX_VALENCE_LIST[_z] = _v
ATOM_MAX_VALENCE = torch.tensor(_ATOM_MAX_VALENCE_LIST, dtype=torch.long)

def compute_node_bond_sum(edge_index, edge_attr, n_nodes):
    """每原子累加入射 bond_order. edge_attr 编码 bond_type: 1/2/3 = 单/双/三, 4 = 芳香键按 1.5 计.

    Returns: [n_nodes] float tensor, 已占价态.
    """
    if edge_index.size(1) == 0:
        return torch.zeros(n_nodes, device=edge_index.device)
    ea = edge_attr.to(torch.float)
    bond_order = torch.where(ea == 4, torch.full_like(ea, 1.5), ea)
    deg = torch.zeros(n_nodes, device=edge_index.device, dtype=bond_order.dtype)
    deg.scatter_add_(0, edge_index[0], bond_order)
    return deg

def compute_partner_max_prob(rc_probs, batch_idx):
    """对每原子 i 返回其所在反应内 **除自身外** 其他原子 rc_prob 的最大值.

    用 top-2 技巧: mol 内全局最大 + 次大, 如果 i 是最大则取次大, 否则取最大.

    Args:
        rc_probs:  [N] float
        batch_idx: [N] long (atom → mol)
    Returns:
        partner_max: [N] float
    """
    device = rc_probs.device
    if rc_probs.numel() == 0:
        return torch.zeros(0, device=device, dtype=rc_probs.dtype)
    n_mols = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0

    mol_max = torch.full((n_mols,), float("-inf"), device=device, dtype=rc_probs.dtype)
    mol_max.scatter_reduce_(0, batch_idx, rc_probs, reduce="amax", include_self=True)
    mol_max = torch.where(torch.isfinite(mol_max), mol_max, torch.zeros_like(mol_max))

    is_max = rc_probs >= (mol_max[batch_idx] - 1e-9)
    rc_probs_without_max = torch.where(
        is_max, torch.full_like(rc_probs, float("-inf")), rc_probs
    )
    mol_2nd = torch.full((n_mols,), float("-inf"), device=device, dtype=rc_probs.dtype)
    mol_2nd.scatter_reduce_(0, batch_idx, rc_probs_without_max, reduce="amax", include_self=True)
    mol_2nd = torch.where(torch.isfinite(mol_2nd), mol_2nd, torch.zeros_like(mol_2nd))

    partner_max = torch.where(is_max, mol_2nd[batch_idx], mol_max[batch_idx])
    return partner_max

def _compute_chunk_weight(chunk_data_list):
    """v15.1: 计算单 chunk 的采样权重（per-reaction 权重均值）

    per-reaction 权重:
      w = 1.0 + 1.5·(n_rc>=5) + 2.0·(mol_size>60) + 1.3·(n_frags>=4), clamp [1, 5]

    动机：v15 ep16 大分子 60-100 atoms F1 81.86%（-12pp vs 20-40 atoms）、多 RC>=5 F1 89.71%、
    多片段 n_frags>=4 F1 85.02%。WeightedRandomSampler 提升这些罕见样本的曝光率。
    """
    weights = []
    for data in chunk_data_list:
        n_atoms = int(data.x.size(0))

        if hasattr(data, "y_delta") and data.y_delta is not None and data.y_delta.dim() == 2 and data.y_delta.numel() > 0:
            n_rc = int(((data.y_delta != 0).any(dim=1) | (data.y_delta != 0).any(dim=0)).sum().item())
        else:
            n_rc = 0

        try:
            if data.edge_index.numel() > 0:
                ei = data.edge_index.cpu().numpy()
                adj = csr_matrix(
                    (np.ones(ei.shape[1], dtype=np.int8), (ei[0], ei[1])),
                    shape=(n_atoms, n_atoms),
                )
                n_frags, _ = connected_components(adj, directed=False)
            else:
                n_frags = n_atoms
        except Exception:
            n_frags = 1

        w = 1.0
        if n_rc >= 5:
            w += 1.5
        if n_atoms > 60:
            w += 2.0
        if n_frags >= 4:
            w += 1.3
        weights.append(min(max(w, 1.0), 5.0))

    return float(np.mean(weights)) if weights else 1.0

def precompute_chunk_weights(files, cache_path=None, verbose=True):
    """v15.1: 扫描所有 chunk 文件一次，计算每个 chunk 的采样权重。

    有 cache 则直接加载；否则全量扫描落盘。~27K chunks × 64 reactions ≈ 3-5 min 一次性开销。
    """
    if cache_path and os.path.exists(cache_path):
        try:
            w = torch.load(cache_path, weights_only=False)
            if isinstance(w, torch.Tensor) and w.numel() == len(files):
                if verbose:
                    print(f"[v15.1] Loaded chunk weights from cache: {cache_path} (size={w.numel()})")
                return w
            else:
                if verbose:
                    print(f"[v15.1] Cache size mismatch, recomputing: cached={w.numel() if isinstance(w, torch.Tensor) else '?'} vs files={len(files)}")
        except Exception as e:
            if verbose:
                print(f"[v15.1] Cache load failed ({type(e).__name__}: {e}), recomputing")

    n = len(files)
    weights = torch.ones(n, dtype=torch.float32)
    t0 = time.time()
    for i, fp in enumerate(files):
        try:
            chunk = torch.load(fp, weights_only=False)
            weights[i] = _compute_chunk_weight(chunk)
        except Exception as e:
            if verbose and i < 5:
                print(f"[v15.1]  chunk {i} {os.path.basename(fp)} failed: {type(e).__name__}: {e}")
            weights[i] = 1.0
        if verbose and (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"[v15.1]  scanned {i+1}/{n} chunks, mean_w={weights[:i+1].mean().item():.3f}, elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    if cache_path:
        try:
            torch.save(weights, cache_path)
            if verbose:
                print(f"[v15.1] Saved chunk weights to {cache_path}")
        except Exception as e:
            print(f"[v15.1]  save failed: {e}")

    if verbose:
        print(f"[v15.1] chunk weights: min={weights.min().item():.3f}, max={weights.max().item():.3f}, "
              f"mean={weights.mean().item():.3f}, >=2={int((weights>=2).sum())}/{n}, "
              f">=3={int((weights>=3).sum())}/{n}")
    return weights

class DiskDataset(Dataset):
    def __init__(self, data_dir, augment=False, data_ratio=1.0):
        self.data_dir = data_dir
        all_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".pt") and f.startswith("chunk_") and "weights" not in f
            ]
        )
        if data_ratio < 1.0:
            num_files = int(len(all_files) * data_ratio)
            self.files = all_files[:num_files]
            print(
                f"Data Subsetting: Using {len(self.files)}/{len(all_files)} files ({data_ratio*100}%)"
            )
        else:
            self.files = all_files
            print(f"Found {len(self.files)} chunk files (Augment={augment})")
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_list = torch.load(self.files[idx], weights_only=False)
        if self.augment:
            new_list = []
            for data in data_list:
                try:

                    if random.random() < 0.5:
                        data = random_permute_graph(data)
                    if random.random() < 0.5:
                        data = random_edge_dropout(data, CONFIG.get("aug_edge_drop_ratio", 0.1))
                    if random.random() < 0.5:
                        data = random_feature_mask(data, CONFIG.get("aug_feat_mask_ratio", 0.15))
                    if random.random() < 0.3:
                        data = random_subgraph_mask(data, CONFIG.get("aug_subgraph_mask_ratio", 0.1))
                except Exception:
                    pass
                new_list.append(data)
            return new_list
        return data_list

def expand_rc_target_by_product_match(
    smi_r, gt_changes_int, rc_target_local, kept_atoms_set,
    max_swap_per_atom=4, skip_high_K=True, K_threshold=4,
    stats=None,
):
    """v29.0 §52 E.B: 对每个 GT RC atom 寻找化学等价位, swap 后产物 canonical SMILES 相同则扩展.

    v29.1 F.A: 加 stats 仪表化, 可选 list[int] of length 8, 函数原地累加.
    stats 维度:
      [0] n_mol_processed   总调用次数
      [1] n_skip_K          K≥K_threshold 跳过
      [2] n_skip_no_gt      无 GT changes 或 K=0
      [3] n_skip_no_mol     RDKit MolFromSmiles / CanonicalRankAtoms 失败
      [4] n_skip_no_canon   _v26_product_canonical 返回空
      [5] n_pos_atoms       GT pos atoms 总数 (in kept_atoms)
      [6] n_swap_attempts   尝试 swap 验证次数
      [7] n_swap_accept     SMILES 一致 → 扩展次数
    派生比率 (eval block 计算):
      skip_K_rate      = [1]/[0]
      swap_accept_rate = [7]/max([6],1)
      expansion_ratio  = [7]/max([5],1)  每 GT pos 平均扩展位数
    """
    if stats is not None:
        stats[0] += 1
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        if not smi_r or not gt_changes_int:
            if stats is not None:
                stats[2] += 1
            return rc_target_local
        K = int((rc_target_local > 0.5).sum().item())
        if skip_high_K and K >= K_threshold:
            if stats is not None:
                stats[1] += 1
            return rc_target_local
        if K == 0:
            if stats is not None:
                stats[2] += 1
            return rc_target_local
        mol = Chem.MolFromSmiles(smi_r)
        if mol is None:
            if stats is not None:
                stats[3] += 1
            return rc_target_local

        try:
            ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        except Exception:
            if stats is not None:
                stats[3] += 1
            return rc_target_local

        gt_smi_canon = _v26_product_canonical(smi_r, gt_changes_int, kept_atoms_set)
        if not gt_smi_canon:
            if stats is not None:
                stats[4] += 1
            return rc_target_local
        expanded = rc_target_local.clone()
        n_atoms = len(rc_target_local)
        pos_atoms = [i for i in range(n_atoms)
                     if i < len(ranks)
                     and rc_target_local[i].item() > 0.5
                     and i in kept_atoms_set]
        if stats is not None:
            stats[5] += len(pos_atoms)
        for pos_i in pos_atoms:
            equiv_candidates = [j for j in range(n_atoms)
                                if j < len(ranks)
                                and ranks[j] == ranks[pos_i]
                                and j != pos_i
                                and j in kept_atoms_set
                                and rc_target_local[j].item() < 0.5]
            for j in equiv_candidates[:max_swap_per_atom]:
                if stats is not None:
                    stats[6] += 1

                swapped_changes = [
                    (j if a == pos_i else a, j if b == pos_i else b, d)
                    for (a, b, d) in gt_changes_int
                ]
                swap_smi_canon = _v26_product_canonical(smi_r, swapped_changes, kept_atoms_set)
                if swap_smi_canon and swap_smi_canon == gt_smi_canon:
                    expanded[j] = 1.0
                    if stats is not None:
                        stats[7] += 1

        return expanded
    except Exception:
        return rc_target_local

def apply_equivalence_class_permutation(data, p=0.5):
    """v29.2 §54 G: 对 mol 内化学等价 atom 随机置换 indices, 保持分子身份不变.

    输入: PyG Data 对象 (data.x / data.edge_index / data.edge_attr / data.y_delta / data.y_dh / data.smiles_r)
    输出: 原地修改 data, 返回 data; 同时 attach data._aug_perm (new_to_old map) 供后续 worker 步骤 remap kept_atoms.

    机制:
      1. Chem.MolFromSmiles(smiles_r) + CanonicalRankAtoms(breakTies=False) → 化学等价类
      2. 在每个等价类内随机洗牌 atom indices → 构造 permutation
      3. 应用 permutation 到所有 atom-level tensors (x, edge_index, y_delta NxN, y_dh)
      4. data._aug_perm 存"new_idx → old_idx"映射, 用于 worker 后续从 SMILES 读 kept_atoms 时 remap

    Permutation-equivariance 保证:
      - GATv2 / AdapterLayer / EdgeT / heads 全是 per-atom/per-edge message passing → 输出自动跟随
      - Loss 是 per-edge BCE / per-atom focal → 对称

    返回 data 不变 (no-op) 的情形:
      - p 抽样不命中
      - smiles_r 缺失或 RDKit 解析失败
      - 等价类全是单原子 (无可交换原子)
    """
    if random.random() > p:
        return data
    smi_r = getattr(data, "smiles_r", None) or getattr(data, "smi_r", None)
    if not smi_r:
        return data
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi_r)
        if mol is None:
            return data
        n_atoms_smi = mol.GetNumAtoms()
        n_atoms_x = data.x.size(0) if data.x is not None else 0
        if n_atoms_smi != n_atoms_x:
            return data
        ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
        from collections import defaultdict
        rank_to_atoms = defaultdict(list)
        for i, r in enumerate(ranks):
            rank_to_atoms[r].append(i)

        nontrivial_classes = [atoms for atoms in rank_to_atoms.values() if len(atoms) > 1]
        if not nontrivial_classes:
            return data

        perm_new_to_old = list(range(n_atoms_x))
        for atoms in nontrivial_classes:
            shuffled = atoms.copy()
            random.shuffle(shuffled)

            for orig_pos, new_atom_at_orig in zip(atoms, shuffled):
                perm_new_to_old[orig_pos] = new_atom_at_orig

        perm_old_to_new = [0] * n_atoms_x
        for new_i, old_i in enumerate(perm_new_to_old):
            perm_old_to_new[old_i] = new_i

        perm_tensor = torch.tensor(perm_new_to_old, dtype=torch.long)

        data.x = data.x[perm_tensor]

        old_to_new_tensor = torch.tensor(perm_old_to_new, dtype=torch.long)
        data.edge_index = old_to_new_tensor[data.edge_index]

        if hasattr(data, "y_delta") and data.y_delta is not None and data.y_delta.dim() == 2:
            data.y_delta = data.y_delta[perm_tensor][:, perm_tensor]

        if hasattr(data, "y_dh") and data.y_dh is not None:
            data.y_dh = data.y_dh[perm_tensor]

        data._aug_perm_new_to_old = perm_new_to_old
        data._aug_perm_old_to_new = perm_old_to_new
        return data
    except Exception:
        return data

def editgnn_collate_eval(batch_list):
    """v29.2 §54 G: val/eval collate, 不应用 data aug.

    必须区分 train/val: train 端 aug 后 atom indices permuted, val 端必须保持 SMILES 原始顺序
    (因为 compute_mol_acc_detailed 用 _v26_product_canonical(smi_r, gt_changes) 需要 gt_changes
    与 smi_r 的 atom indices 对齐).
    """
    return editgnn_collate(batch_list, apply_aug=False)

def editgnn_collate(batch_list, apply_aug=True):
    """EditGNN collate: y_delta 存为 list，不做 NxN padding

    速度优化: 在 worker 进程中预计算 frag_id / tokenization，
    避免主线程 forward() 中的 CPU 瓶颈（scipy connected_components 等）。

    v29.2 §54 G: 入口处应用 equivalence_class data aug (按 CONFIG.equiv_aug_prob).
    val_loader 应使用 editgnn_collate_eval (apply_aug=False) 避免 SMILES atom-index mismatch.
    """
    if not batch_list:
        return None
    data_list = [item for chunk in batch_list for item in chunk]
    if not data_list:
        return None

    y_delta_list = []
    y_dh_list = []
    env_x_list = []
    env_batch_list = []
    text_list = []

    aug_enabled = apply_aug and CONFIG.get("use_equivalence_class_aug", False)
    aug_prob = float(CONFIG.get("equiv_aug_prob_active", CONFIG.get("equiv_aug_prob", 0.0)))

    for i, data in enumerate(data_list):

        if aug_enabled and aug_prob > 0.0:
            data = apply_equivalence_class_permutation(data, p=aug_prob)

        y_delta_list.append(data.y_delta)
        del data.y_delta

        if hasattr(data, "y_dh") and data.y_dh is not None:
            y_dh_list.append(data.y_dh)
            del data.y_dh
        else:

            y_dh_list.append(torch.zeros(data.x.size(0), dtype=torch.long))

        if hasattr(data, "env_x") and data.env_x is not None:
            env_x_list.append(data.env_x)
            env_batch_list.append(
                torch.full((data.env_x.size(0),), i, dtype=torch.long)
            )
        else:
            env_x_list.append(torch.zeros((1, 1024)))
            env_batch_list.append(torch.tensor([i], dtype=torch.long))

        proc = getattr(data, "proc_text", "")
        work = getattr(data, "work_text", "")
        text_list.append(f"{proc} [SEP] {work}")

    big_batch = Batch.from_data_list(data_list)
    batch_size = len(data_list)

    big_batch.env_x_all = torch.cat(env_x_list, dim=0)
    big_batch.env_batch = torch.cat(env_batch_list, dim=0)

    if TOKENIZER:

        cached_ids = []
        cached_masks = []
        all_cached = True
        for txt in text_list:
            if txt in _TOKEN_CACHE:
                ids, mask = _TOKEN_CACHE[txt]
                cached_ids.append(ids)
                cached_masks.append(mask)
            else:
                all_cached = False
                break

        if all_cached:

            max_len = max(ids.size(0) for ids in cached_ids)
            padded_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
            padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
            for i, (ids, mask) in enumerate(zip(cached_ids, cached_masks)):
                L = ids.size(0)
                padded_ids[i, :L] = ids
                padded_masks[i, :L] = mask
            big_batch.input_ids = padded_ids
            big_batch.attention_mask = padded_masks
        else:
            tokenized = TOKENIZER(
                text_list,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            big_batch.input_ids = tokenized["input_ids"]
            big_batch.attention_mask = tokenized["attention_mask"]

            for i, txt in enumerate(text_list):
                if txt not in _TOKEN_CACHE:
                    mask = tokenized["attention_mask"][i]
                    seq_len = mask.sum().item()
                    _TOKEN_CACHE[txt] = (
                        tokenized["input_ids"][i, :seq_len].clone(),
                        mask[:seq_len].clone(),
                    )
    else:
        big_batch.input_ids = torch.zeros((batch_size, 10), dtype=torch.long)
        big_batch.attention_mask = torch.zeros((batch_size, 10), dtype=torch.long)

    big_batch.y_delta_list = y_delta_list
    big_batch.y_dh_list = y_dh_list

    if CONFIG.get("use_role_signals", False) and _extract_role_signals is not None:
        d_role = int(CONFIG.get("role_signal_dim", 43))
        role_signals = torch.zeros(batch_size, d_role, dtype=torch.float32)
        n_precomputed = 0
        for i, data in enumerate(data_list):

            pre = getattr(data, "role_signals", None)
            if pre is not None and isinstance(pre, torch.Tensor) and pre.numel() == d_role:
                role_signals[i] = pre
                n_precomputed += 1
                continue

            global _ROLE_SIGNAL_CACHE
            if len(_ROLE_SIGNAL_CACHE) > _ROLE_SIGNAL_CACHE_MAX:
                _ROLE_SIGNAL_CACHE.clear()
            rxn_id = getattr(data, "rxn_id", None)
            if rxn_id is not None and rxn_id in _ROLE_SIGNAL_CACHE:
                cached = _ROLE_SIGNAL_CACHE[rxn_id]
                if cached.shape[0] == d_role:
                    role_signals[i] = cached
                    continue
            try:
                feat = _extract_role_signals(
                    smiles_r=getattr(data, "smiles_r", "") or "",
                    env_smiles_list=getattr(data, "env_smiles_list", None) or [],
                    proc_text=getattr(data, "proc_text", "") or "",
                    work_text=getattr(data, "work_text", "") or "",
                    ab_flags=getattr(data, "ab_flags", None),
                )
            except Exception:
                feat = torch.zeros(d_role, dtype=torch.float32)
            role_signals[i] = feat
            if rxn_id is not None:
                _ROLE_SIGNAL_CACHE[rxn_id] = feat
        big_batch.role_signals = role_signals
    else:
        big_batch.role_signals = None

    big_batch.smi_r_list = [getattr(d, "smiles_r", "") or "" for d in data_list]
    big_batch.smi_p_list = [getattr(d, "smiles_p", "") or "" for d in data_list]

    rc_sym_enabled = CONFIG.get("use_rc_symmetry_expansion", False)
    rc_sym_max_swap = int(CONFIG.get("rc_sym_max_swap_per_atom", 4))
    rc_sym_skip_high = bool(CONFIG.get("rc_sym_skip_high_K", True))
    rc_sym_skip_K = int(CONFIG.get("rc_sym_skip_K_threshold", 4))

    rc_sym_stats_enabled = CONFIG.get("use_rc_sym_stats_logging", False) and rc_sym_enabled
    rc_sym_stats_list = [0] * 8 if rc_sym_stats_enabled else None
    _CLS_TO_DELTA_EXP = {1: +1, 2: -1, 3: +2, 4: -2, 5: +3, 6: -3}
    try:
        rc_targets = []
        pair_chunks = []
        atom_offset = 0
        for i_mol, yd in enumerate(y_delta_list):
            row_active = (yd != 0).any(dim=1)
            col_active = (yd != 0).any(dim=0)
            rc_tgt_local = (row_active | col_active).float()

            if rc_sym_enabled:
                try:
                    smi_r_i = getattr(data_list[i_mol], "smiles_r", "") or ""
                    if smi_r_i:
                        gt_changes_int = []
                        if yd.dim() == 2 and yd.numel() > 0:
                            triu_pairs = torch.triu(yd != 0, diagonal=1).nonzero(as_tuple=False).tolist()
                            for ij in triu_pairs:
                                cls = int(yd[ij[0], ij[1]].item())
                                if cls in _CLS_TO_DELTA_EXP:
                                    gt_changes_int.append((int(ij[0]), int(ij[1]), _CLS_TO_DELTA_EXP[cls]))
                        if gt_changes_int:

                            from rdkit import Chem as _ChemExp
                            from rdkit import RDLogger as _RDLog
                            _RDLog.DisableLog("rdApp.*")
                            r_mol = _ChemExp.MolFromSmiles(smi_r_i)
                            if r_mol is not None:
                                kept_set = {a.GetIdx() for a in r_mol.GetAtoms()
                                            if a.GetAtomMapNum() > 0}
                                if kept_set:
                                    rc_tgt_local = expand_rc_target_by_product_match(
                                        smi_r_i, gt_changes_int, rc_tgt_local, kept_set,
                                        max_swap_per_atom=rc_sym_max_swap,
                                        skip_high_K=rc_sym_skip_high,
                                        K_threshold=rc_sym_skip_K,
                                        stats=rc_sym_stats_list,
                                    )
                except Exception:
                    pass
            rc_targets.append(rc_tgt_local)
            if yd.dim() == 2 and yd.numel() > 0:
                triu = torch.triu(yd != 0, diagonal=1)
                pairs_local = triu.nonzero(as_tuple=False)
                if pairs_local.numel() > 0:
                    pair_chunks.append(pairs_local + atom_offset)
            atom_offset += yd.size(0) if yd.dim() == 2 else 0
        big_batch.precomputed_rc_target_flat = torch.cat(rc_targets)
        big_batch.precomputed_rc_pair_index = (
            torch.cat(pair_chunks).long() if pair_chunks
            else torch.empty((0, 2), dtype=torch.long)
        )

        if rc_sym_stats_list is not None:
            big_batch.rc_sym_stats = torch.tensor(rc_sym_stats_list, dtype=torch.long)
    except Exception:
        pass

    try:
        atom_offset_main = 0
        main_masks = []

        for i, data in enumerate(data_list):
            n_local = data.x.size(0)
            local_ei = data.edge_index
            local_ea = data.edge_attr.float() if data.edge_attr is not None else torch.zeros(0)
            yd_local = y_delta_list[i]
            mm = compute_main_product_atoms(local_ei, local_ea, yd_local, n_local)
            main_masks.append(mm)
            atom_offset_main += n_local
        big_batch.main_atom_mask = torch.cat(main_masks) if main_masks else torch.zeros(0, dtype=torch.bool)
    except Exception:
        big_batch.main_atom_mask = None

    if hasattr(big_batch, "y_yield"):
        big_batch.y_yield = big_batch.y_yield.view(batch_size)

    try:
        num_atoms = big_batch.x.size(0)
        gfid, lfid, n_gf = compute_frag_ids_global(
            big_batch.edge_index, big_batch.batch, num_atoms
        )
        big_batch.precomputed_global_frag_id = gfid
        big_batch.precomputed_local_frag_id = lfid

        big_batch.precomputed_num_global_frags = torch.tensor(n_gf, dtype=torch.long)
    except Exception:
        pass

    try:
        num_atoms = big_batch.x.size(0)
        if num_atoms > 0 and big_batch.edge_index.size(1) > 0:
            src_np = big_batch.edge_index[0].numpy()
            dst_np = big_batch.edge_index[1].numpy()
            adj_ones = np.ones(len(src_np), dtype=np.float32)
            adj_sparse_full = csr_matrix(
                (adj_ones, (src_np, dst_np)), shape=(num_atoms, num_atoms)
            )

            dist_full = shortest_path(
                adj_sparse_full, method="D", directed=False, unweighted=True,
            )

            dist_int8 = np.where(np.isinf(dist_full), 8, dist_full)
            dist_int8 = np.clip(dist_int8, 0, 127).astype(np.int8)
            big_batch.precomputed_hop_matrix = dist_int8
        else:
            big_batch.precomputed_hop_matrix = np.zeros(
                (num_atoms, num_atoms), dtype=np.int8
            )
    except Exception:
        pass

    return big_batch

class AtomFeatureEncoder(nn.Module):
    def __init__(self, hidden_dim, feature_dims):
        super().__init__()
        self.embeddings = nn.ModuleList()
        self.feature_dims = feature_dims
        for dim in feature_dims:
            self.embeddings.append(nn.Embedding(dim, hidden_dim))

    def forward(self, x):
        out = 0
        for i in range(len(self.embeddings)):
            feat_col = x[:, i]
            max_idx = self.feature_dims[i] - 1
            feat_col_clamped = feat_col.clamp(0, max_idx)
            if i == 0:
                out = self.embeddings[i](feat_col_clamped)
            else:
                out = out + self.embeddings[i](feat_col_clamped)
        return out

class MoleculeEncoder(nn.Module):
    """6层 GATv2, 无 JK-Net — 匹配预训练 checkpoint"""

    def __init__(
        self, hidden_dim=512, num_layers=6, num_heads=8, edge_embedding_dim=64
    ):
        super().__init__()
        self.atom_encoder = AtomFeatureEncoder(hidden_dim, ATOM_FEATURE_DIMS)
        self.edge_embedding = nn.Embedding(10, edge_embedding_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_embedding_dim,
                )
            )
            self.norms.append(LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(0.25)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_encoder(x)
        edge_attr = edge_attr.clamp(0, 9)
        edge_emb = self.edge_embedding(edge_attr)

        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index, edge_attr=edge_emb)
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = x + x_in
        return x

class ElectronicPretrainingModel(nn.Module):
    def __init__(self, encoder, hidden_dim=512):
        super().__init__()
        self.encoder = encoder
        self.fukui_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.att_pool = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        )
        self.homo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )
        self.lumo_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, edge_index, edge_attr, batch_idx):
        """v10: batch_idx 的语义被外部调用方决定。
        - 预训练阶段: batch_idx == reaction batch index (每个分子是一个 batch 元素，
          天然就是 fragment 级)
        - 反应训练阶段 (EditGNNModel): 调用方传入 global_frag_id, 让 graph pooling
          按 fragment 而非 reaction 聚合 (修复了之前 multi-mol reaction 共享 HOMO/LUMO 的 bug)
        新增返回 graph_emb 以便 FragmentDescriptor 复用 att_pool 学到的图级表征。
        """
        node_emb = self.encoder(x, edge_index, edge_attr)
        fukui = self.fukui_head(node_emb)
        charge = self.charge_head(node_emb)
        graph_emb = self.att_pool(node_emb, batch_idx)
        homo = self.homo_head(graph_emb)
        lumo = self.lumo_head(graph_emb)
        return node_emb, fukui, charge, homo, lumo, graph_emb

class EnvironmentEncoder(nn.Module):
    def __init__(self, fp_dim=1024, hidden_dim=512):
        super().__init__()
        self.molecule_embedding = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, batch_first=True
        )
        self.pooling_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, env_x, env_batch):
        x_dense, mask = to_dense_batch(env_x, env_batch)
        padding_mask = ~mask
        h = self.molecule_embedding(x_dense)

        h_ctx, _ = self.attention(h, h, h, key_padding_mask=padding_mask)
        h_ctx = torch.nan_to_num(h_ctx, 0.0)
        B = h.size(0)
        query = self.pooling_query.expand(B, -1, -1)
        v_env, _ = self.attention(query, h_ctx, h_ctx, key_padding_mask=padding_mask)
        v_env = torch.nan_to_num(v_env, 0.0)
        return v_env.squeeze(1)

class TextEncoder(nn.Module):
    def __init__(self, model_name, hidden_dim=512):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        self.projection = nn.Linear(768, hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projection(cls_token)

class ConditionEncoder(nn.Module):
    def __init__(self, hidden_dim=512):
        super().__init__()
        self.temp_emb = nn.Embedding(8, 32)
        self.mlp = nn.Sequential(
            nn.Linear(37, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, temp_id, time_val, ab_flags):
        if temp_id.dim() == 0:
            temp_id = temp_id.unsqueeze(0)
        elif temp_id.dim() > 1:
            temp_id = temp_id.view(-1)
        t_emb = self.temp_emb(temp_id.long())
        if time_val.dim() == 1:
            time_val = time_val.unsqueeze(-1)
        if ab_flags.dim() == 1:
            ab_flags = ab_flags.unsqueeze(0)
        cat_feat = torch.cat([t_emb, time_val, ab_flags], dim=-1)
        return self.mlp(cat_feat)

def load_pretrained_model(path, device):
    """加载预训练 checkpoint (6层GATv2, 无JK)"""
    print(f"Loading Pretrained Elec-Model from {path}...")
    enc = MoleculeEncoder(
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["gnn_layers"],
        num_heads=CONFIG["gnn_heads"],
    )
    model = ElectronicPretrainingModel(enc, hidden_dim=CONFIG["hidden_dim"])
    try:
        sd = torch.load(path, map_location=device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        new_sd = {
            k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()
        }
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            print(f"  Missing keys: {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys: {unexpected[:5]}...")
        print(f"  Weights loaded successfully.")
    except Exception as e:
        print(f"  Load failed: {e}. Using Random Init.")
    return model

class AdapterLayer(nn.Module):
    """冻结 encoder 后的可训练适配层 (~262K params)"""

    def __init__(self, hidden_dim=512, bottleneck_dim=256, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = F.gelu(x)
        x = self.drop(self.up(x))
        return x + residual

class DeepAdapterLayer(nn.Module):
    """双层 Adapter: 更大容量，用于 RC 适配（~590K params）

    v4 RC 专攻: 单层 bottleneck=256 表达力不足，无法区分
    「参与反应的碳」和「不参与反应的碳」等细粒度差异。
    双层结构通过两次非线性变换提升特征空间的区分度。
    """

    def __init__(self, hidden_dim=512, bottleneck_dim=384, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.down1 = nn.Linear(hidden_dim, bottleneck_dim)
        self.up1 = nn.Linear(bottleneck_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.down2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.up2 = nn.Linear(bottleneck_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm1(x)
        h = self.drop1(self.up1(F.gelu(self.down1(h)))) + x
        h = self.drop2(self.up2(F.gelu(self.down2(self.norm2(h))))) + h
        return h

def compute_frag_ids_global(edge_index, batch_idx, num_nodes):
    """对整个 batch 计算原子的连通分量 ID。

    Args:
        edge_index: [2, E] 整个 batch 的边索引（全局原子下标）
        batch_idx: [N] 每个原子所属反应 ID
        num_nodes: int

    Returns:
        global_frag_id: [N] long, 跨整个 batch 唯一的分子 ID
            (用于按分子做 graph pooling，得到每个 fragment 一组 HOMO/LUMO)
        local_frag_id:  [N] long, 同一反应内分子的局部编号 0..K-1，clamp 到 7
            (用于 frag_emb / FragmentDescriptor 索引)
        num_global_frags: int, 全局分子数
    """
    device = batch_idx.device
    if num_nodes == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, 0

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    if len(src) == 0:

        labels = np.arange(num_nodes, dtype=np.int64)
    else:
        ones = np.ones(len(src), dtype=np.int8)
        adj = csr_matrix((ones, (src, dst)), shape=(num_nodes, num_nodes))
        _, labels = connected_components(adj, directed=False)
        labels = labels.astype(np.int64)

    batch_cpu = batch_idx.cpu()
    labels_t = torch.from_numpy(labels)
    max_comp = int(labels.max()) + 1 if len(labels) > 0 else 1
    combined = batch_cpu * max_comp + labels_t
    unique_combined, inverse = combined.unique(return_inverse=True)
    num_global_frags = unique_combined.numel()

    global_frag = inverse.to(device)

    batch_size = int(batch_cpu.max()) + 1 if batch_cpu.numel() > 0 else 1
    node_counts = torch.bincount(batch_cpu, minlength=batch_size)
    node_offsets = torch.zeros(batch_size, dtype=torch.long)
    node_offsets[1:] = node_counts[:-1].cumsum(0)
    mol_first_inv = inverse[node_offsets]
    local_frag = (inverse - mol_first_inv[batch_cpu]).clamp(max=7).to(device)

    return global_frag, local_frag, num_global_frags

class CrossFragmentAttention(nn.Module):
    """多层跨片段注意力: 让同一反应内不同分子片段的原子深度交换信息。

    v1 只有 1 层 4 头注意力，对于需要多步推理的分子间反应（如 SN2 亲核进攻
    需要同时理解亲核试剂的 HOMO 和底物的 LUMO 位点）信息交换不充分。

    v2 改进:
    1. 多层 Transformer block (默认 2 层)，每层含 attention + FFN + residual
    2. 更多注意力头 (8 头)，捕捉更丰富的分子间交互模式
    3. 分子角色编码: 通过连通分量 ID 为不同分子添加位置偏置，
       让模型区分"同一分子内远距离原子"和"不同分子的原子"

    v10 改动:
    - 不再持有 frag_emb (nn.Embedding(8, D))，分子角色编码改由 FragmentDescriptor
      提供 (基于结构/物理量)。fragment 上下文向量从外部传入并直接广播加到原子上。
    - 支持外部传入预先算好的 local_frag_ids，避免重复 connected_components 计算。

    v11 改动:
    - FragmentDescriptor 输出改为门控残差: graph_emb + gate*(proj - graph_emb)，
      初期 gate≈0.05 输出≈预训练 graph_emb，不污染 h_cross；随训练自动打开。
    """

    def __init__(self, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers

        self.attn_layers = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm1_layers = nn.ModuleList()
        self.norm2_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
            )
            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
            )
            self.norm1_layers.append(nn.LayerNorm(hidden_dim))
            self.norm2_layers.append(nn.LayerNorm(hidden_dim))

    def forward(self, h, batch_idx, frag_context=None):
        """
        h: [N_total, hidden_dim] 所有原子嵌入
        batch_idx: [N_total] 每个原子属于哪个反应
        frag_context: [N_total, hidden_dim] (可选) 每个原子对应的 fragment 描述符
            (来自 FragmentDescriptor)，作为跨分子角色编码加到原子表征上

        Returns: [N_total, hidden_dim]
        """
        if h.size(0) == 0:
            return h

        h_padded, mask = to_dense_batch(h, batch_idx)
        B, max_N, D = h_padded.shape

        mem_estimate = B * max_N * D * 12
        if mem_estimate > 1.5e9 or max_N > 500:
            return h

        if frag_context is not None:
            h_padded[mask] = h_padded[mask] + frag_context * 0.3

        padding_mask = ~mask

        x = h_padded
        for i in range(self.num_layers):
            if self.training:
                def custom_forward(
                    module_attn, module_norm1, module_ffn, module_norm2, inp, mask
                ):
                    attn_out, _ = module_attn(inp, inp, inp, key_padding_mask=mask)
                    attn_out = torch.nan_to_num(attn_out, 0.0)
                    out = module_norm1(inp + attn_out)
                    out = module_norm2(out + module_ffn(out))
                    return out

                x = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    self.attn_layers[i],
                    self.norm1_layers[i],
                    self.ffn_layers[i],
                    self.norm2_layers[i],
                    x,
                    padding_mask,
                    use_reentrant=False,
                )
            else:
                h_attn, _ = self.attn_layers[i](
                    x, x, x, key_padding_mask=padding_mask
                )
                h_attn = torch.nan_to_num(h_attn, 0.0)
                x = self.norm1_layers[i](x + h_attn)
                x = self.norm2_layers[i](x + self.ffn_layers[i](x))

        return x[mask]

HALIDE_ATOMIC_NUMS = {9, 17, 35, 53}
HETERO_ATOMIC_NUMS = {7, 8, 15, 16, 9, 17, 35, 53}

class FragmentDescriptor(nn.Module):
    """v10: 用结构 + 物理量为每个分子片段构建描述符。

    替代原来的 nn.Embedding(8, D) frag 编号编码 (那个仅区分"是第几个分子",
    没有携带任何化学语义)。本模块的输出可作为:
    1. CrossFragmentAttention 中跨分子角色编码 (加到原子表征上)
    2. 任何下游需要"per-fragment 上下文"的模块

    输入信号 (per-fragment, 全部从已有特征聚合):
        - HOMO / LUMO / gap            (3)  电子学语义，决定 FMO 反应性
        - mean Fukui (f+ / f-)         (2)  亲核 / 亲电位点平均强度
        - total formal charge          (1)  分子总电荷
        - log(n_heavy)                 (1)  分子大小先验
        - aromatic ratio               (1)  芳香性 (来自 atom 特征 col 4)
        - halide count / n_heavy       (1)  含 F/Cl/Br/I 比例 (离去基/亲电信号)
        - hetero ratio                 (1)  N/O/P/S + halide 比例
        - max degree                   (1)  最高配位数 (代表 sp3 中心 vs 桥头)
        - att_pool graph_emb           (D)  GNN 学到的图级表征
    总维度: 11 + D, 通过 2 层 MLP 投影到 D。
    """

    def __init__(self, hidden_dim=512):
        super().__init__()

        self.fukui_channels = tuple(CONFIG.get("frag_desc_fukui_channels", ["mean"]))
        extra_fukui_dim = 0
        if "max" in self.fukui_channels:
            extra_fukui_dim += 2
        if "top3" in self.fukui_channels:
            extra_fukui_dim += 2
        in_dim = 11 + extra_fukui_dim + hidden_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        nn.init.constant_(self.gate_mlp[-1].bias, -3.0)
        nn.init.zeros_(self.gate_mlp[-1].weight)

    @staticmethod
    def _scatter_mean(values, index, num_segments):
        """简单的 scatter_mean: values [N, D] index [N] -> out [num_segments, D]"""
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        out = torch.zeros(num_segments, values.size(-1), device=values.device, dtype=values.dtype)
        cnt = torch.zeros(num_segments, 1, device=values.device, dtype=values.dtype)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(values), values)
        cnt.scatter_add_(0, index.unsqueeze(-1), torch.ones_like(index.unsqueeze(-1), dtype=values.dtype))
        return out / (cnt + 1e-8)

    @staticmethod
    def _scatter_sum(values, index, num_segments):
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        out = torch.zeros(num_segments, values.size(-1), device=values.device, dtype=values.dtype)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(values), values)
        return out

    @staticmethod
    def _scatter_max(values, index, num_segments):
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        out = torch.full((num_segments, values.size(-1)), float("-inf"),
                         device=values.device, dtype=values.dtype)
        out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(values), values, reduce="amax", include_self=True)
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

    def forward(self, x_atom_raw, fukui, charge, edge_index, global_frag_id,
                num_global_frags, homo, lumo, graph_emb):
        """
        x_atom_raw: [N, 8] 原子离散特征 (Z, degree, formal_charge, hyb, arom, mass, ch, h)
        fukui:      [N, 2]
        charge:     [N, 1]
        edge_index: [2, E] 用于 max degree
        global_frag_id: [N] 每个原子对应的全局 fragment ID
        num_global_frags: int
        homo / lumo:    [num_global_frags, 1]
        graph_emb:      [num_global_frags, D]

        Returns: [num_global_frags, D]
        """
        device = x_atom_raw.device
        if num_global_frags == 0:
            return torch.empty((0, graph_emb.size(-1)), device=device)

        atomic_num = x_atom_raw[:, 0].long()
        is_aromatic = (x_atom_raw[:, 4] > 0).float().unsqueeze(-1)
        is_halide = torch.zeros_like(is_aromatic)
        for z in HALIDE_ATOMIC_NUMS:
            is_halide = is_halide + (atomic_num == z).float().unsqueeze(-1)
        is_hetero = torch.zeros_like(is_aromatic)
        for z in HETERO_ATOMIC_NUMS:
            is_hetero = is_hetero + (atomic_num == z).float().unsqueeze(-1)

        n_heavy = self._scatter_sum(torch.ones(x_atom_raw.size(0), 1, device=device),
                                    global_frag_id, num_global_frags)
        log_n_heavy = torch.log1p(n_heavy)

        mean_fukui = self._scatter_mean(fukui.float(), global_frag_id, num_global_frags)

        extra_fukui_channels = []
        if "max" in self.fukui_channels:
            max_fukui = self._scatter_max(fukui.float(), global_frag_id, num_global_frags)
            extra_fukui_channels.append(max_fukui)
        if "top3" in self.fukui_channels:

            extra_fukui_channels.append(mean_fukui)
        total_charge = self._scatter_sum(charge.float(), global_frag_id, num_global_frags)
        aromatic_ratio = self._scatter_mean(is_aromatic, global_frag_id, num_global_frags)
        halide_ratio = self._scatter_mean(is_halide, global_frag_id, num_global_frags)
        hetero_ratio = self._scatter_mean(is_hetero, global_frag_id, num_global_frags)

        num_atoms = x_atom_raw.size(0)
        if edge_index.size(1) > 0:
            atom_degree = torch.bincount(edge_index[0], minlength=num_atoms).float()
        else:
            atom_degree = torch.zeros(num_atoms, device=device)
        atom_degree = atom_degree.clamp(max=15.0).unsqueeze(-1)
        max_degree = self._scatter_max(atom_degree, global_frag_id, num_global_frags)

        gap = F.softplus(lumo - homo) + 0.5

        feat = torch.cat([
            homo, lumo, gap,
            mean_fukui,
            *extra_fukui_channels,
            total_charge,
            log_n_heavy,
            aromatic_ratio,
            halide_ratio,
            hetero_ratio,
            max_degree,
            graph_emb,
        ], dim=-1)
        raw = self.proj(feat)

        gate = torch.sigmoid(self.gate_mlp(torch.cat([graph_emb, raw], dim=-1)))
        return graph_emb + gate * (raw - graph_emb)

class EdgeRCHead(nn.Module):
    """v10 Phase 1 升级: 键级反应中心辅助头

    动机（来自 cc_pair_failures_v9_ep30.md）:
      C-C pair 失败中 neither (双漏) 占 48%, half (半漏) 占 52%。
      原子级 RCHead 学到了"亲电那一侧"但漏掉对端，本质是缺少"键级监督信号"。
      EdgeRCHead 直接对每条**真分子键** (batch.edge_index) 二分类——这条键是否参与反应。

    设计要点:
      - 不替换原 atom-level RCHead，纯辅助 loss，反向传播改善共享的 h_rc 表征
      - 推理时不直接使用其输出（避免改决策接口）
      - 输入: [h_i, h_j, |h_i - h_j|, h_i*h_j, edge_emb]，~ 4*D + 8 = 2056 维
      - 轻量 2 层 MLP，参数 ~ 1.2M
      - 真键标签来自 y_delta_list (compute_loss 中构建)
    """

    def __init__(self, hidden_dim=512, num_bond_types=8, dropout=0.15):
        super().__init__()
        self.bond_emb = nn.Embedding(num_bond_types, 8)
        in_dim = 4 * hidden_dim + 8
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h, edge_index, edge_attr):
        """
        Args:
            h: [N, D]   atom representations (post-cross_frag, pre-adapter is fine)
            edge_index: [2, E]  分子图真边 (双向)
            edge_attr:  [E]     bond type 索引 (1=single, 2=double, ...)
        Returns:
            bond_logits_dir: [E] 每条**有向边**的 logit
                上层用 (i<j) mask 取无向后再算 loss / metric
        """
        if edge_index.size(1) == 0:
            return torch.empty((0,), device=h.device, dtype=h.dtype)
        src, dst = edge_index[0], edge_index[1]
        h_i = h[src]
        h_j = h[dst]
        diff = (h_i - h_j).abs()
        prod = h_i * h_j
        bt = self.bond_emb(edge_attr.long().clamp(min=0, max=self.bond_emb.num_embeddings - 1))
        feat = torch.cat([h_i, h_j, diff, prod, bt], dim=-1)
        return self.mlp(feat).squeeze(-1)

class PairFragmentAttention(nn.Module):
    """v13: pair-level per-reaction self-attention.

    输入 pair_emb [P, D] + pair_group_id [P]，按 group 分组做 dense MHA + FFN。
    原子级 FragmentFullAttention 的 pair-level 对应物，targeted 解 C-C pair both_rate 82.07%。
    显存安全: 单组 pair 数 > max_pairs_per_group 时跳过整个 batch 的 attention。
    """

    def __init__(self, hidden_dim=512, num_heads=4, max_pairs_per_group=150):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.max_pairs_per_group = max_pairs_per_group

    def forward(self, pair_emb, pair_group_id):
        """
        pair_emb: [P, D] — 每条候选 pair / refinement edge 的表征
        pair_group_id: [P] — 每条 pair 的分组 (通常是 reaction_id 或 frag_id)
        """
        if pair_emb.numel() == 0 or pair_emb.size(0) < 2:
            return pair_emb

        sorted_idx = pair_group_id.argsort(stable=True)
        emb_sorted = pair_emb[sorted_idx]
        grp_sorted = pair_group_id[sorted_idx]

        emb_padded, mask = to_dense_batch(emb_sorted, grp_sorted)
        G_batch, max_P, D = emb_padded.shape

        if max_P > self.max_pairs_per_group or max_P < 2:
            return pair_emb

        padding_mask = ~mask
        attn_out, _ = self.mha(
            emb_padded, emb_padded, emb_padded, key_padding_mask=padding_mask
        )
        attn_out = torch.nan_to_num(attn_out, 0.0)
        emb_padded = self.norm1(emb_padded + attn_out)
        emb_padded = self.norm2(emb_padded + self.ffn(emb_padded))

        emb_out_sorted = emb_padded[mask]
        inv_idx = torch.empty_like(sorted_idx)
        inv_idx[sorted_idx] = torch.arange(pair_emb.size(0), device=pair_emb.device)
        return emb_out_sorted[inv_idx]

class RCPairRefiner(nn.Module):
    """v11: RC-aware Pairwise Refinement — 在 RC 候选配对图上精修 rc_logits
    v15: 移除 v14 的 Round 2（edge_mlp2/update_mlp2）+ Confidence-Weighted Scatter

    解决 half-pair 的三类失败模式:
      - A (intra_break, 53%): 分子内键两端通过 edge_index 直接传消息
      - B (inter_form, 37%):  跨分子 top-K RC 候选互相传消息
      - C (intra_form, 10%):  同分子 top-K 非邻居对传消息

    核心优势: 推理时直接生效（不是辅助 loss），用 RC 概率作为边特征实现 pair-level 推理。
    单一 loss: 标准 BCE 作用于 rc_logits_final，梯度自然反传到 RCPairRefiner + RCHead。

    信息流:
      Round 1: rc_logits_raw = RCHead(h_rc, ...)
      refinement_graph = build(edge_index + top-K cross-pairs)
      delta_logit = message_passing(h_rc, rc_probs_raw, refinement_graph)
      rc_logits_final = rc_logits_raw + gate * delta_logit
    """

    def __init__(self, hidden_dim=512):
        super().__init__()

        edge_in = hidden_dim * 3 + 4
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
        )

        self.pair_frag_attn = PairFragmentAttention(hidden_dim)

        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, 512),
            nn.GELU(),
            nn.Linear(512, 1),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.gate[-1].bias, -3.0)
        nn.init.zeros_(self.gate[-1].weight)

    def build_refinement_graph(self, rc_probs, edge_index, batch_idx, frag_id, top_k=6):
        """构建精修图: 分子内键 + 跨分子 RC 候选对 + 分子内远程对

        Args:
            rc_probs: [N_total] 初始 RC 概率
            edge_index: [2, E] 分子图真边 (双向)
            batch_idx: [N_total] 每个原子所属反应
            frag_id: [N_total] 全局 fragment ID
            top_k: 每个反应取 top-K 个 RC 候选

        Returns:
            ref_ei: [2, E'] 精修图边 (双向)
            same_frag: [E'] bool, True=同一 fragment
            has_bond: [E'] bool, True=分子内已有键
        """
        device = rc_probs.device

        src_list = [edge_index[0]]
        dst_list = [edge_index[1]]
        n_type0 = edge_index.size(1)

        probs_dense, probs_mask = to_dense_batch(rc_probs, batch_idx)
        frag_dense, _ = to_dense_batch(frag_id, batch_idx)
        B, max_N = probs_dense.shape

        atoms_per_rxn = probs_mask.sum(dim=1)
        k_per_rxn = torch.clamp(atoms_per_rxn, max=top_k)

        probs_dense[~probs_mask] = float("-inf")
        k_max = min(top_k, max_N)

        if k_max >= 2:

            topk_vals, topk_local = probs_dense.topk(k_max, dim=1)

            counts = torch.bincount(batch_idx, minlength=B)
            offsets = torch.zeros(B, dtype=torch.long, device=device)
            offsets[1:] = counts[:-1].cumsum(0)
            topk_global = topk_local + offsets[:, None]
            topk_frags = frag_dense.gather(1, topk_local)

            pair_a = []
            pair_b = []
            for a in range(k_max):
                for b in range(a + 1, k_max):
                    pair_a.append(a)
                    pair_b.append(b)
            if pair_a:
                pair_a_t = torch.tensor(pair_a, device=device)
                pair_b_t = torch.tensor(pair_b, device=device)
                n_pairs = len(pair_a)

                src_idx = topk_global[:, pair_a_t]
                dst_idx = topk_global[:, pair_b_t]
                src_frags = topk_frags[:, pair_a_t]
                dst_frags = topk_frags[:, pair_b_t]
                same_frag_pairs = (src_frags == dst_frags)

                valid_rxn = k_per_rxn >= 2

                pair_b_expanded = pair_b_t.unsqueeze(0).expand(B, -1)
                valid_pair = (pair_b_expanded < k_per_rxn[:, None]) & valid_rxn[:, None]

                valid_flat = valid_pair.reshape(-1)
                src_flat = src_idx.reshape(-1)[valid_flat]
                dst_flat = dst_idx.reshape(-1)[valid_flat]
                sf_flat = same_frag_pairs.reshape(-1)[valid_flat]

                extra_src_t = torch.cat([src_flat, dst_flat])
                extra_dst_t = torch.cat([dst_flat, src_flat])
                extra_sf_t = torch.cat([sf_flat, sf_flat])

                src_list.append(extra_src_t)
                dst_list.append(extra_dst_t)
            else:
                extra_sf_t = torch.empty(0, dtype=torch.bool, device=device)
        else:
            extra_sf_t = torch.empty(0, dtype=torch.bool, device=device)

        ref_src = torch.cat(src_list)
        ref_dst = torch.cat(dst_list)
        ref_ei = torch.stack([ref_src, ref_dst])

        sf_type0 = torch.ones(n_type0, dtype=torch.bool, device=device)
        hb_type0 = torch.ones(n_type0, dtype=torch.bool, device=device)

        hb_extra = torch.zeros(len(extra_sf_t), dtype=torch.bool, device=device)

        same_frag_all = torch.cat([sf_type0, extra_sf_t]) if len(extra_sf_t) > 0 else sf_type0
        has_bond_all = torch.cat([hb_type0, hb_extra]) if len(extra_sf_t) > 0 else hb_type0

        return ref_ei, same_frag_all, has_bond_all

    def forward(self, h_rc, rc_probs, edge_index, batch_idx, frag_id, top_k=6):
        """
        Args:
            h_rc: [N_total, D] adapter_rc 输出的原子表征
            rc_probs: [N_total] Round 1 RC 概率 (不 detach，允许端到端梯度)
            edge_index: [2, E] 分子图真边
            batch_idx: [N_total] 反应 ID
            frag_id: [N_total] 全局 fragment ID
            top_k: 每反应取 top-K RC 候选

        Returns:
            delta_logit: [N_total] 对每个原子的 rc_logit 修正量 (已乘 gate)
        """
        ref_ei, same_frag, has_bond = self.build_refinement_graph(
            rc_probs, edge_index, batch_idx, frag_id, top_k
        )

        if ref_ei.size(1) == 0:
            return torch.zeros(h_rc.size(0), device=h_rc.device)

        src, dst = ref_ei

        h_i, h_j = h_rc[src], h_rc[dst]
        p_i = rc_probs[src].unsqueeze(-1)
        p_j = rc_probs[dst].unsqueeze(-1)
        sf = same_frag.float().unsqueeze(-1)
        hb = has_bond.float().unsqueeze(-1)

        edge_feat = torch.cat([h_i, h_j, (h_i - h_j).abs(), p_i, p_j, sf, hb], dim=-1)
        msg = self.edge_mlp(edge_feat)

        pair_rxn_id = batch_idx[src]
        msg = self.pair_frag_attn(msg, pair_rxn_id)

        N = h_rc.size(0)
        agg = torch.zeros(N, msg.size(1), device=h_rc.device, dtype=msg.dtype)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)
        count = torch.zeros(N, 1, device=h_rc.device, dtype=msg.dtype)
        ones = torch.ones(dst.size(0), 1, device=h_rc.device, dtype=msg.dtype)
        count.scatter_add_(0, dst.unsqueeze(-1), ones)
        agg = agg / count.clamp(min=1.0)

        gate = torch.sigmoid(self.gate(torch.cat([h_rc, rc_probs.unsqueeze(-1)], dim=-1)))
        delta_logit = self.update_mlp(torch.cat([h_rc, agg, rc_probs.unsqueeze(-1)], dim=-1))
        delta_r1 = (gate * delta_logit).squeeze(-1)

        return delta_r1

class _FragFullAttnLayer(nn.Module):
    """v15: 单层 FragmentFullAttention block (MHA + norm + FFN + norm2)

    从 v14 的 FragmentFullAttention 拆出，方便多层堆叠。
    第 2 层 identity-init 由外部 warmstart 逻辑负责（out_proj/ffn 末层置零）。
    """

    def __init__(self, hidden_dim=512, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, h_padded, padding_mask):
        attn_out, _ = self.mha(
            h_padded, h_padded, h_padded, key_padding_mask=padding_mask
        )
        attn_out = torch.nan_to_num(attn_out, 0.0)
        h_padded = self.norm(h_padded + attn_out)
        h_padded = self.norm2(h_padded + self.ffn(h_padded))
        return h_padded

class FragmentFullAttention(nn.Module):
    """v12: 片段内全图注意力 — 让同一分子的所有原子互相 attend
    v15: 堆叠到 num_layers 层（默认 2 层），攻 bulk C FN (占 53%) + C-C pair both 82% + 大分子长程交互

    按 frag_id (global_frag_id) 分组做 dense MHA，避免跨分子混淆。
    显存安全: 片段 > max_frag_atoms 时自动跳过。

    v14 → v15 迁移：
    - v14 直接存 mha/norm/ffn/norm2 在类上（1 层）
    - v15 改用 layers[0].mha / layers[0].norm / ... 形式，warmstart 需要 key rename
    - 为保持 warmstart 兼容，__init__ 时若 num_layers==1 退化到旧路径；默认 2 层
    """

    def __init__(self, hidden_dim=512, num_heads=4, max_frag_atoms=300, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [_FragFullAttnLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.max_frag_atoms = max_frag_atoms

    def forward(self, h, frag_id):

        sorted_idx = frag_id.argsort(stable=True)
        h_sorted = h[sorted_idx]
        frag_sorted = frag_id[sorted_idx]

        h_padded, mask = to_dense_batch(h_sorted, frag_sorted)
        F_batch, max_N, D = h_padded.shape

        if max_N > self.max_frag_atoms:
            return h

        padding_mask = ~mask
        for layer in self.layers:
            h_padded = layer(h_padded, padding_mask)

        h_out_sorted = h_padded[mask]
        inv_idx = torch.empty_like(sorted_idx)
        inv_idx[sorted_idx] = torch.arange(h.size(0), device=h.device)
        return h_out_sorted[inv_idx]

class GlobalAnchorAttention(nn.Module):
    """v16: Per-fragment learnable anchor tokens for cross-atom global long-range signal.

    动机：v12-v15.1 四代共 17 次迭代验证，bulk C FN 在 17-21K 间固化（占总 FN 53%+），
    Top 30 FN 物化特征正常（fukui/charge 与 TP 几乎不可区分），纯粹缺跨片段长程信号。
    FragFullAttn 2 层 + max_frag_atoms=400 仍不够（v15.1 ep17 C-C both 80.29% 反向退步）。

    机制：每个 fragment 分配 K 个 learnable anchor tokens（共享参数），通过双向 cross-attention
    在 O(N·K) 复杂度下建立原子↔全局语义通路，绕开 MHA O(N²) 瓶颈：
        Phase A: anchors ← atoms  (anchor 从 fragment 所有原子聚合信息)
        Phase B: atoms   ← anchors (atom 从 anchors 取回压缩的全局信号)

    K=4 << N≤400，显存增量约 +1GB。与 FragFullAttn 级联后，每个原子可通过 anchor
    跨更远距离获取 context（相当于 1-hop 到 anchor，K 次转发）。
    """

    def __init__(self, d_model=512, n_anchors=4, n_heads=8, dropout=0.1, max_frag_atoms=400):
        super().__init__()
        self.n_anchors = n_anchors
        self.max_frag_atoms = max_frag_atoms

        self.anchor_embed = nn.Parameter(torch.randn(n_anchors, d_model) * 0.02)

        self.atom_to_anchor = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.anchor_to_atom = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln_a = nn.LayerNorm(d_model)
        self.ln_x = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(self, h, frag_id):
        """h: [N_total, D], frag_id: [N_total] -> [N_total, D]"""
        sorted_idx = frag_id.argsort(stable=True)
        h_sorted = h[sorted_idx]
        frag_sorted = frag_id[sorted_idx]
        x_pad, mask = to_dense_batch(h_sorted, frag_sorted)
        B, max_N, D = x_pad.shape

        if max_N > self.max_frag_atoms:
            return h

        key_padding_mask = ~mask

        anchors = self.anchor_embed.unsqueeze(0).expand(B, -1, -1)

        anchors_upd, _ = self.atom_to_anchor(
            query=anchors, key=x_pad, value=x_pad,
            key_padding_mask=key_padding_mask,
        )
        anchors_upd = torch.nan_to_num(anchors_upd, 0.0)
        anchors = self.ln_a(anchors + anchors_upd)

        x_upd, _ = self.anchor_to_atom(query=x_pad, key=anchors, value=anchors)
        x_upd = torch.nan_to_num(x_upd, 0.0)
        x_pad = self.ln_x(x_pad + x_upd)
        x_pad = self.ln_ffn(x_pad + self.ffn(x_pad))

        h_out_sorted = x_pad[mask]
        inv_idx = torch.empty_like(sorted_idx)
        inv_idx[sorted_idx] = torch.arange(h.size(0), device=h.device)
        return h_out_sorted[inv_idx]

class PairSymmetryHead(nn.Module):
    """v16: 显式 pair-level shared prob head, 攻 C-C pair 对称性崩溃。

    动机：v11 → v15.1 共 5 代，C-C pair both_rate 在 80-84% 震荡（v15.1 ep17 80.29% 新低），
    half_rate 7% 固化（一端极高 prob + 另一端 0）。v13 pair_sym / PairFragAttn /
    v14 Two-Round Refinement / v15 pair_contrastive 全部失败 — loss 层软约束 + attention 层
    增容均无效。v16 切换到 pair-level 显式 head，直接输出 shared prob_pair。

    训练时两条 loss：
      1. Pair BCE: 对候选 pair 做 bond/non-bond 二分类
      2. Pair → Atom teacher: 对真 RC pair, 强制 prob_atom_i ≈ prob_atom_j ≈ prob_pair.detach()

    输入特征: [h_i, h_j, h_i * h_j, |h_i - h_j|] 捕捉 atom embedding 对称/不对称信号。
    """

    def __init__(self, d_model=512, hidden=256, dropout=0.1):
        super().__init__()
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_model * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_i, h_j):
        """h_i, h_j: [P, D] -> pair_logits [P]."""
        feat = torch.cat([h_i, h_j, h_i * h_j, (h_i - h_j).abs()], dim=-1)
        return self.pair_mlp(feat).squeeze(-1)

class PairHardHead(nn.Module):
    """v18 N1: 独立 pair binary head，hard supervision from y_delta[i,j] != 0。

    攻根本原因 6 (Phase 1 唯一 hard target 是 rc_target atom binary，pair 全是 soft 约束)。
    v16 PairSymHead 的 supervision 通过 atom prob teacher 间接产生，v16.1 H1 不得不改 asymmetric
    margin 规避反向污染。PairHardHead 用 (y_delta[i,j] != 0) 作硬标签，独立 head 独立梯度，
    绕开 atom prob 中间层。

    输入特征: [h_i, h_j, |h_i - h_j|] (3D)，比 PairSymHead 少一个 h_i*h_j (4D)，
    因为硬监督已足够驱动对称性学习，元素积易过参数化。
    """

    def __init__(self, d_model=512, hidden=256, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, h_atom, pair_index):
        """
        h_atom:     [N, D] — RCHead 输出的 h_for_pair（post-GlobalAnchor atom hidden）
        pair_index: [2, P] or [P, 2] — batch-global atom indices (i, j)
        return:     [P] pair logits
        """
        if pair_index.numel() == 0:
            return h_atom.new_zeros(0)
        if pair_index.dim() == 2 and pair_index.size(0) == 2 and pair_index.size(1) != 2:
            i, j = pair_index[0], pair_index[1]
        else:

            i, j = pair_index[:, 0], pair_index[:, 1]
        h_i, h_j = h_atom[i], h_atom[j]
        feat = torch.cat([h_i, h_j, (h_i - h_j).abs()], dim=-1)
        return self.mlp(feat).squeeze(-1)

class PairConsistencyGate(nn.Module):
    """v19 V1 (§32.3): PairHardHead pair logit → atom-level rc_logit 残差抬升.

    - 仅抬升（ReLU boost + α ≥ 0），不降低 (D3 合规)
    - max 聚合：救 FN 只需一个高置信 partner
    - α learnable + warmup schedule（set_alpha_from_warmup 挂到 epoch loop）
    """

    def __init__(self, config):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.alpha_target = float(config.get("pair_cons_alpha_target", 0.3))
        self.use_relu_boost = bool(config.get("pair_cons_use_relu", True))

    def forward(self, rc_logits, pair_hard_logits, pair_hard_index, n_atoms):
        """
        rc_logits:        [N]
        pair_hard_logits: [P] or None
        pair_hard_index:  [2, P] 或 [P, 2] batch-global atom indices
        n_atoms:          int (= rc_logits.size(0))
        Returns: rc_logit_final [N]
        """
        if pair_hard_logits is None or pair_hard_logits.numel() == 0:
            return rc_logits

        boost_src = pair_hard_logits
        if self.use_relu_boost:
            boost_src = F.relu(boost_src)

        if pair_hard_index.dim() == 2 and pair_hard_index.size(0) == 2 and pair_hard_index.size(1) != 2:
            i_idx = pair_hard_index[0]
            j_idx = pair_hard_index[1]
        else:
            i_idx = pair_hard_index[:, 0]
            j_idx = pair_hard_index[:, 1]

        zeros = torch.zeros(n_atoms, device=rc_logits.device, dtype=rc_logits.dtype)
        boost_src = boost_src.to(zeros.dtype)
        pair_boost_i = zeros.scatter_reduce(0, i_idx, boost_src, reduce="amax", include_self=True)
        pair_boost_j = zeros.scatter_reduce(0, j_idx, boost_src, reduce="amax", include_self=True)
        pair_boost = torch.maximum(pair_boost_i, pair_boost_j)

        return rc_logits + self.alpha * pair_boost

    def set_alpha_from_warmup(self, epoch, warmup_start, warmup_end):
        """线性 ramp α: [warmup_start, warmup_end] → [0, alpha_target]"""
        if warmup_end <= warmup_start:
            self.alpha.data.fill_(self.alpha_target)
            return
        if epoch < warmup_start:
            self.alpha.data.fill_(0.0)
        elif epoch >= warmup_end:
            self.alpha.data.fill_(self.alpha_target)
        else:
            r = (epoch - warmup_start) / (warmup_end - warmup_start)
            self.alpha.data.fill_(r * self.alpha_target)

class FukuiProjection(nn.Module):
    """v19 V3 (§32.5): Fukui(原值+frag-内 delta) 专属 projection."""

    def __init__(self, d_out=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, fukui, fukui_delta):
        return self.mlp(torch.cat([fukui, fukui_delta], dim=-1))

class ChargeProjection(nn.Module):
    """v19 V3 (§32.5): Charge 专属 projection."""

    def __init__(self, d_out=32, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, d_out),
        )

    def forward(self, charge):
        return self.mlp(charge)

class FMOProjection(nn.Module):
    """v19 V4 (§32.6): HOMO/LUMO/gap 原子级 projection."""

    def __init__(self, d_out=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
            nn.Linear(32, d_out),
        )

    def forward(self, homo_atom, lumo_atom, gap_atom):
        return self.mlp(torch.cat([homo_atom, lumo_atom, gap_atom], dim=-1))

class RLCFiLMConditioner(nn.Module):
    """v20 M1: 用 v_global (含 text+env+cond) 生成 atom-level FiLM 参数 (γ, β)。

    γ ⊙ h + β,把反应级语境直接调制每个原子的特征。
    比 attention conditioning 更直接,参数更省。
    """
    def __init__(self, atom_dim, r_dim=512, dropout=0.1):
        super().__init__()
        self.mlp_gamma = nn.Sequential(
            nn.Linear(r_dim, atom_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(atom_dim, atom_dim),
        )
        self.mlp_beta = nn.Sequential(
            nn.Linear(r_dim, atom_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(atom_dim, atom_dim),
        )
        nn.init.zeros_(self.mlp_gamma[-1].weight)
        nn.init.zeros_(self.mlp_gamma[-1].bias)
        nn.init.zeros_(self.mlp_beta[-1].weight)
        nn.init.zeros_(self.mlp_beta[-1].bias)

    def forward(self, h_atom, v_global, batch_idx):
        """
        h_atom:    [N, D]
        v_global:  [B, r_dim]
        batch_idx: [N] long, 原子所属反应索引

        Returns: h_atom_cond = (1 + γ) ⊙ h_atom + β  [N, D]
                 (γ 残差形式确保 init=0 时恒等映射,warmstart 安全)
        """
        v_per_atom = v_global[batch_idx]
        gamma = self.mlp_gamma(v_per_atom)
        beta = self.mlp_beta(v_per_atom)
        return (1.0 + gamma) * h_atom + beta

class RoleSignalProjector(nn.Module):
    """v25 §48.2 路线 A: 把 40 维 SMARTS binary 信号投影到 hidden_dim,
    残差加到 v_global, 经 RLCFiLMConditioner 调制每原子.

    末层 init=0 → warmstart 时输出 0, v_global 不变, 与 v24 best.pth 恒等接续.
    训练后 init 偏离 0, role_signals 才开始影响 atom-level FiLM.
    """
    def __init__(self, in_dim=40, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, role_signals):
        """role_signals: [B, in_dim] float → [B, hidden_dim] float."""
        return self.proj(role_signals)

class PerFragRcCalibrator(nn.Module):
    """v20 M3: frag 内 z-score 归一化 + 全局可学温度/偏置。

    把 ranking 强但阈值化弱的"排得对但 prob 不够过 0.5"的 RC 推过阈值。
    init: T=3.0 (对应 v19 ep21 logit 标准差估计), b=0.0 → 启动时近似恒等(scale only)。
    """
    def __init__(self, init_T=3.0, init_b=0.0, eps=1e-5, min_frag=3):
        super().__init__()
        self.T = nn.Parameter(torch.tensor(float(init_T)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.eps = float(eps)
        self.min_frag = int(min_frag)

    def forward(self, rc_logits, frag_id, num_frags):
        """
        rc_logits: [N] float
        frag_id:   [N] long, 全局 frag 索引(batch 内 unique)
        num_frags: int

        Returns: rc_logits_calibrated [N]
        """
        device = rc_logits.device
        N = rc_logits.size(0)
        if N == 0 or num_frags == 0:
            return rc_logits

        frag_sum = torch.zeros(num_frags, device=device, dtype=rc_logits.dtype)
        frag_count = torch.zeros(num_frags, device=device, dtype=rc_logits.dtype)
        frag_sum.scatter_add_(0, frag_id, rc_logits)
        frag_count.scatter_add_(0, frag_id, torch.ones_like(rc_logits))
        frag_count_safe = frag_count.clamp_min(1.0)
        frag_mean = frag_sum / frag_count_safe

        diff = rc_logits - frag_mean[frag_id]
        frag_sq_sum = torch.zeros(num_frags, device=device, dtype=rc_logits.dtype)
        frag_sq_sum.scatter_add_(0, frag_id, diff * diff)
        frag_var = frag_sq_sum / frag_count_safe
        frag_std = (frag_var + self.eps).sqrt()

        normalize_mask = (frag_count >= float(self.min_frag))
        std_per_atom = frag_std[frag_id]
        mean_per_atom = frag_mean[frag_id]
        mask_per_atom = normalize_mask[frag_id]

        normalized = (rc_logits - mean_per_atom) / std_per_atom

        rc_logits_z = torch.where(mask_per_atom, normalized, rc_logits)

        offset = torch.where(mask_per_atom, mean_per_atom, torch.zeros_like(mean_per_atom))
        rc_logits_calibrated = self.T * rc_logits_z + self.b + offset
        return rc_logits_calibrated

def compute_mol_aux_loss(
    rc_probs,
    rc_target_flat,
    batch_idx,
    temperature_atom=0.02,
    clean_threshold=0.95,
):
    """v20 M4 MLA Phase 1 版: 反应级 clean 软指示作为辅助 BCE。

    score_atom_correct_i = 1 - |rc_prob_i - rc_target_i|       # ∈ [0,1]
    score_mol_m = mean_{i ∈ mol_m}(score_atom_correct_i)
    P_clean_m = σ((score_mol_m - τ) / temperature)
    is_clean_m = (mean(score_atom_correct_i ≥ τ_atom) == 1.0) 严格分子全对
                 实际用 hard label: all atoms correct under thr=0.5

    返回 BCE(P_clean, is_clean_hard).
    Args:
        rc_probs:        [N] float in [0,1]
        rc_target_flat:  [N] float ∈ {0,1}
        batch_idx:       [N] long
        temperature_atom: float, sigmoid 温度
        clean_threshold: float, score_mol 阈值 τ
    """
    if rc_probs.numel() == 0 or rc_target_flat is None:
        return torch.tensor(0.0, device=rc_probs.device)

    device = rc_probs.device
    n_mols = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
    if n_mols == 0:
        return torch.tensor(0.0, device=device)

    score_atom = 1.0 - (rc_probs - rc_target_flat).abs()

    pred_hard = (rc_probs >= 0.5).float()
    correct_hard = (pred_hard == rc_target_flat).float()
    mol_correct_count = torch.zeros(n_mols, device=device, dtype=score_atom.dtype)
    mol_total = torch.zeros(n_mols, device=device, dtype=score_atom.dtype)
    mol_correct_count.scatter_add_(0, batch_idx, correct_hard)
    mol_total.scatter_add_(0, batch_idx, torch.ones_like(correct_hard))
    is_clean_hard = (mol_correct_count >= mol_total).float()

    mol_score_sum = torch.zeros(n_mols, device=device, dtype=score_atom.dtype)
    mol_score_sum.scatter_add_(0, batch_idx, score_atom)
    mol_total_safe = mol_total.clamp_min(1.0)
    score_mol = mol_score_sum / mol_total_safe

    logit_mol = (score_mol - clean_threshold) / max(temperature_atom, 1e-6)
    P_clean = torch.sigmoid(logit_mol)
    P_clean = P_clean.clamp(1e-7, 1.0 - 1e-7)

    loss = -(is_clean_hard * torch.log(P_clean)
             + (1.0 - is_clean_hard) * torch.log(1.0 - P_clean)).mean()
    return loss

class RCHead(nn.Module):
    """原子级二分类: 是否为反应中心

    v8 增强 (目标 97% F1):
    1. 5-hop JK-GAT + 交替跨分子注意力:
       - 5 层 GATv2Conv 覆盖长共轭链 (α,β-不饱和羰基 ~4 hop, 苯环共轭 ~6 hop)
       - JK-Attention 避免过平滑: 每个原子学习从 6 个尺度 (0-5 hop) 选择最佳感受野
       - 每 2 层 GAT 后插入 1 层跨分子注意力，刷新分子间信息
         GAT1→GAT2→CrossAttn→GAT3→GAT4→CrossAttn→GAT5
         解决 v7 的核心问题: 跨分子信息只注入一次后被 5 层分子内 GAT 逐步稀释
    2. 门控条件融合: v_global 通过学习的 gate 与原子特征交互
    3. 加宽 MLP: 1603 → 1024 → 512 → 256 → 1 + skip connection
    4. 度数嵌入加宽: 32 → 64
    """

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.num_hops = 5

        self.cross_attn_after = {1, 3}

        self.neighbor_attns = nn.ModuleList()
        self.neighbor_norms = nn.ModuleList()
        for _ in range(self.num_hops):
            self.neighbor_attns.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // 4,
                    heads=4,
                    concat=True,
                    add_self_loops=False,
                )
            )
            self.neighbor_norms.append(nn.LayerNorm(hidden_dim))

        self.inter_mol_attns = nn.ModuleList()
        self.inter_mol_norms = nn.ModuleList()
        for _ in range(len(self.cross_attn_after)):
            self.inter_mol_attns.append(
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=4,
                    dropout=0.1,
                    batch_first=True,
                )
            )
            self.inter_mol_norms.append(nn.LayerNorm(hidden_dim))

        self.jk_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1, bias=False),
        )

        self.degree_emb = nn.Embedding(16, 64)

        self.cond_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

        self.frag_full_attn = FragmentFullAttention(
            hidden_dim,
            num_layers=CONFIG.get("frag_full_attn_layers", 2),
            max_frag_atoms=CONFIG.get("max_frag_atoms", 300),
        )

        if CONFIG.get("use_global_anchor", False):
            n_ga_layers = CONFIG.get("n_global_anchor_layers", 1)
            self.global_anchor = nn.ModuleList([
                GlobalAnchorAttention(
                    d_model=hidden_dim,
                    n_anchors=CONFIG.get("n_global_anchors", 4),
                    n_heads=CONFIG.get("global_anchor_heads", 8),
                    dropout=CONFIG.get("global_anchor_dropout", 0.1),
                    max_frag_atoms=CONFIG.get("max_frag_atoms", 400),
                )
                for _ in range(n_ga_layers)
            ])
        else:
            self.global_anchor = None

        self.use_fukui_projection = bool(CONFIG.get("use_fukui_projection", False))
        self.use_fmo_atom_injection = bool(CONFIG.get("use_fmo_atom_injection", False))

        if self.use_fukui_projection:
            fukui_proj_dim = int(CONFIG.get("fukui_proj_dim", 64))
            charge_proj_dim = int(CONFIG.get("charge_proj_dim", 32))
            fukui_drop = float(CONFIG.get("fukui_proj_dropout", 0.1))
            self.fukui_proj = FukuiProjection(d_out=fukui_proj_dim, dropout=fukui_drop)
            self.charge_proj = ChargeProjection(d_out=charge_proj_dim, dropout=fukui_drop)
        else:
            fukui_proj_dim = 2
            charge_proj_dim = 1
            self.fukui_proj = None
            self.charge_proj = None

        if self.use_fmo_atom_injection:
            fmo_proj_dim = int(CONFIG.get("fmo_proj_dim", 32))
            self.fmo_proj = FMOProjection(d_out=fmo_proj_dim)
        else:
            fmo_proj_dim = 0
            self.fmo_proj = None

        feat_dim = hidden_dim * 4 + fukui_proj_dim + charge_proj_dim + fmo_proj_dim + 64
        rc_drop = CONFIG.get("rc_head_dropout", 0.2)

        self.fc1 = nn.Linear(feat_dim, 1024)
        self.norm1 = nn.LayerNorm(1024)
        self.drop1 = nn.Dropout(rc_drop)
        self.fc2 = nn.Linear(1024, 512)
        self.norm2 = nn.LayerNorm(512)
        self.drop2 = nn.Dropout(rc_drop)
        self.fc3 = nn.Linear(512, 256)
        self.norm3 = nn.LayerNorm(256)
        self.drop3 = nn.Dropout(rc_drop)
        self.skip_proj = nn.Linear(feat_dim, 256)
        self.fc_out = nn.Linear(256, 1)

        self.reaction_bias = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.reaction_bias[-1].bias, 0.0)

    def _inter_mol_cross_attn(self, h, batch_idx, attn_idx):
        """轻量跨分子注意力: 让同一反应内不同分子的原子交换信息

        Args:
            h: [N_total, D] 当前原子表征
            batch_idx: [N_total] 每个原子所属反应 ID
            attn_idx: int, 使用第几个 cross-attention module
        Returns:
            [N_total, D] 融合了跨分子信息的表征
        """
        h_padded, mask = to_dense_batch(h, batch_idx)
        B, max_N, D = h_padded.shape

        mem_estimate = B * max_N * D * 8
        if mem_estimate > 1.0e9 or max_N > 500:
            return h

        padding_mask = ~mask
        attn_out, _ = self.inter_mol_attns[attn_idx](
            h_padded, h_padded, h_padded, key_padding_mask=padding_mask
        )
        attn_out = torch.nan_to_num(attn_out, 0.0)
        h_padded = self.inter_mol_norms[attn_idx](h_padded + attn_out)

        return h_padded[mask]

    def forward(self, h_adapted, fukui, charge, edge_index, batch_idx, v_global,
                frag_id=None, return_h_for_pair=False,
                global_frag_id=None, num_global_frags=None,
                homo_frag=None, lumo_frag=None):
        """
        h_adapted: [total_nodes, 512]
        fukui: [total_nodes, 2]
        charge: [total_nodes, 1]
        edge_index: [2, E] 分子图拓扑
        batch_idx: [total_nodes] 每个原子属于哪个反应
        v_global: [B, 512] 全局反应条件
        frag_id: [total_nodes] v12 每个原子的 fragment ID (用于片段全图注意力)
        return_h_for_pair: v16 — 若为 True 额外返回 post-GlobalAnchor 的 atom hidden (供 PairSymmetryHead 使用)

        v19 新增（§32.5/32.6）:
            global_frag_id:   [total_nodes] long — 全局 fragment ID（V3 fukui frag-内 delta 需要）
            num_global_frags: int
            homo_frag:        [num_global_frags, 1] — V4 frag 级 HOMO（会广播到原子）
            lumo_frag:        [num_global_frags, 1] — V4 frag 级 LUMO
        当 use_fukui_projection/use_fmo_atom_injection 关闭时，新参数被忽略，行为等价 v18。
        """

        layer_outputs = [h_adapted]
        h = h_adapted
        cross_idx = 0
        for i in range(self.num_hops):
            h_new = self.neighbor_attns[i](h, edge_index)
            h = self.neighbor_norms[i](h_new + h)

            if i in self.cross_attn_after:
                h = self._inter_mol_cross_attn(h, batch_idx, cross_idx)
                cross_idx += 1
            layer_outputs.append(h)

        stacked = torch.stack(layer_outputs, dim=1)
        attn_scores = self.jk_attn(stacked)
        attn_weights = torch.softmax(attn_scores, dim=1)
        h_jk = (attn_weights * stacked).sum(dim=1)

        if frag_id is not None:
            h_jk = self.frag_full_attn(h_jk, frag_id)

        if self.global_anchor is not None and frag_id is not None:
            for ga_layer in self.global_anchor:
                h_jk = ga_layer(h_jk, frag_id)

        h_for_pair = h_jk

        num_nodes = h_adapted.size(0)
        degree = torch.bincount(edge_index[0], minlength=num_nodes)[:num_nodes].clamp(
            max=15
        )
        deg_feat = self.degree_emb(degree)

        v_global_expanded = v_global[batch_idx]
        gate = self.cond_gate(torch.cat([h_adapted, v_global_expanded], dim=-1))
        h_cond_fused = h_adapted + gate * self.cond_proj(v_global_expanded)

        h_cross = h_adapted * h_jk

        if self.use_fukui_projection and self.fukui_proj is not None:
            if global_frag_id is not None and num_global_frags is not None and num_global_frags > 0:

                fukui_f = fukui.to(torch.float32)
                gfid = global_frag_id.to(torch.long)
                ng = int(num_global_frags)

                sum_per_frag = torch.zeros(ng, fukui_f.size(-1), device=fukui_f.device, dtype=fukui_f.dtype)
                cnt_per_frag = torch.zeros(ng, 1, device=fukui_f.device, dtype=fukui_f.dtype)
                sum_per_frag.scatter_add_(0, gfid.unsqueeze(-1).expand_as(fukui_f), fukui_f)
                cnt_per_frag.scatter_add_(0, gfid.unsqueeze(-1), torch.ones_like(gfid, dtype=fukui_f.dtype).unsqueeze(-1))
                mean_per_frag = sum_per_frag / (cnt_per_frag + 1e-8)
                fukui_mean_at_atom = mean_per_frag[gfid]
                fukui_delta = (fukui_f - fukui_mean_at_atom).to(fukui.dtype)
            else:
                fukui_delta = torch.zeros_like(fukui)
            fukui_feat = self.fukui_proj(fukui, fukui_delta)
            charge_feat = self.charge_proj(charge)
        else:
            fukui_feat = fukui
            charge_feat = charge

        if (self.use_fmo_atom_injection and self.fmo_proj is not None
                and homo_frag is not None and lumo_frag is not None
                and global_frag_id is not None):
            gfid = global_frag_id.to(torch.long)
            homo_atom = homo_frag[gfid]
            lumo_atom = lumo_frag[gfid]
            gap_atom = F.softplus(lumo_atom - homo_atom) + 0.5
            fmo_feat = self.fmo_proj(homo_atom, lumo_atom, gap_atom)
            extra = [fmo_feat]
        else:
            extra = []

        x = torch.cat(
            [h_adapted, h_jk, h_cond_fused, fukui_feat, charge_feat] + extra + [deg_feat, h_cross],
            dim=-1,
        )

        skip = self.skip_proj(x)
        h = self.drop1(F.gelu(self.norm1(self.fc1(x))))
        h = self.drop2(F.gelu(self.norm2(self.fc2(h))))
        h = self.drop3(F.gelu(self.norm3(self.fc3(h))))
        h = h + skip
        logits = self.fc_out(h).squeeze(-1)

        rxn_bias = self.reaction_bias(v_global)
        logits = logits + rxn_bias[batch_idx].squeeze(-1)

        if return_h_for_pair:
            return logits, h_for_pair
        return logits

def extract_rc_target(y_delta):
    """从 y_delta 矩阵提取 RC 标签 (哪些原子参与反应)"""

    row_active = (y_delta != 0).any(dim=1)
    col_active = (y_delta != 0).any(dim=0)
    return (row_active | col_active).float()

def extract_rc_pairs_from_y_delta(y_delta):
    """v10 Phase 1: 从 y_delta 中提取无序 (i, j) 反应键对索引。

    用于 pair-symmetric loss 与 bond-aware RC 标签。
    返回 LongTensor [P, 2] (i < j)，P 可能为 0。
    """
    if y_delta is None or y_delta.numel() == 0 or y_delta.dim() != 2:
        return torch.empty((0, 2), dtype=torch.long)
    triu_mask = torch.triu(y_delta != 0, diagonal=1)
    pairs = triu_mask.nonzero(as_tuple=False)
    return pairs.long()

def build_candidate_edges_cpu(
    edge_index_np, edge_attr_np, num_nodes, rc_mask_np, y_delta_np=None, rc_all_pairs=False,
    rc_mask_soft_np=None, atom_local_frag_id_np=None,
):
    """
    构建候选边列表 — 纯 CPU numpy 版本

    对 ~30 原子的小分子，CPU numpy 操作比 GPU tensor 快 10-100x
    （GPU 的 kernel launch 开销 ~10μs/op，而 numpy 30×30 操作 ~100ns）

    Args:
            edge_index_np: [2, E] numpy int64
            edge_attr_np: [E] numpy int64
            num_nodes: int
            rc_mask_np: [N] numpy bool — 硬阈值 RC mask
            y_delta_np: [N, N] numpy int64 or None
            rc_all_pairs: bool
            rc_mask_soft_np: [N] numpy bool or None — v10 软阈值 RC mask（包含 hard 全集 + 额外低置信原子）
                仅用于生成跨分子 RC×RC pair（推理覆盖率安全网）
            atom_local_frag_id_np: [N] numpy int64 or None — 每个原子的分子内 fragment ID
                soft 跨分子 pair 仅在 frag_id_i != frag_id_j 时生成

    Returns:
            cand_edges: [K, 2] numpy int64 (i < j)
            cand_bond_types: [K] numpy int64
    """
    empty_edges = np.empty((0, 2), dtype=np.int64)
    empty_bt = np.empty((0,), dtype=np.int64)

    if edge_index_np.shape[1] == 0:
        if y_delta_np is not None:
            triu = np.triu(y_delta_np, k=1)
            nz = np.argwhere(triu != 0)
            if len(nz) > 0:
                return nz.astype(np.int64), np.zeros(len(nz), dtype=np.int64)
        return empty_edges, empty_bt

    src, dst = edge_index_np[0], edge_index_np[1]

    fwd = src < dst
    exist_i = src[fwd]
    exist_j = dst[fwd]
    exist_bt = edge_attr_np[fwd]

    adj = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    adj[src, dst] = True

    rc_idx = np.where(rc_mask_np)[0]
    extra_i_list = []
    extra_j_list = []

    if len(rc_idx) > 0:

        rc_neigh = adj[rc_idx]
        pairs = np.argwhere(rc_neigh)
        if len(pairs) > 0:
            rn_i = rc_idx[pairs[:, 0]]
            rn_j = pairs[:, 1]
            rn_min = np.minimum(rn_i, rn_j)
            rn_max = np.maximum(rn_i, rn_j)
            valid = rn_min < rn_max
            if valid.any():
                extra_i_list.append(rn_min[valid])
                extra_j_list.append(rn_max[valid])

        if len(rc_idx) > 1:
            if rc_all_pairs:
                R = len(rc_idx)
                row_idx, col_idx = np.triu_indices(R, k=1)
                extra_i_list.append(rc_idx[row_idx])
                extra_j_list.append(rc_idx[col_idx])
            else:
                adj2 = (adj.astype(np.float32) @ adj.astype(np.float32)) > 0
                reachable = adj | adj2
                rc_reach = reachable[np.ix_(rc_idx, rc_idx)]
                np.fill_diagonal(rc_reach, False)
                rc_pairs = np.argwhere(np.triu(rc_reach, k=1))
                if len(rc_pairs) > 0:
                    extra_i_list.append(rc_idx[rc_pairs[:, 0]])
                    extra_j_list.append(rc_idx[rc_pairs[:, 1]])

    if (
        rc_mask_soft_np is not None
        and atom_local_frag_id_np is not None
        and rc_mask_soft_np.any()
    ):

        rc_union = rc_mask_np | rc_mask_soft_np
        union_idx = np.where(rc_union)[0]
        if len(union_idx) > 1:
            R = len(union_idx)
            row_idx, col_idx = np.triu_indices(R, k=1)
            ui = union_idx[row_idx]
            uj = union_idx[col_idx]

            cross_mol_mask = atom_local_frag_id_np[ui] != atom_local_frag_id_np[uj]
            if cross_mol_mask.any():
                extra_i_list.append(ui[cross_mol_mask])
                extra_j_list.append(uj[cross_mol_mask])

    all_i = np.concatenate([exist_i] + extra_i_list) if extra_i_list else exist_i
    all_j = np.concatenate([exist_j] + extra_j_list) if extra_j_list else exist_j

    if y_delta_np is not None:
        gt_nz = np.argwhere(np.triu(y_delta_np, k=1) != 0)
        if len(gt_nz) > 0:
            all_i = np.concatenate([all_i, gt_nz[:, 0]])
            all_j = np.concatenate([all_j, gt_nz[:, 1]])

    if len(all_i) == 0:
        return empty_edges, empty_bt

    edge_ids = all_i * num_nodes + all_j
    unique_ids = np.unique(edge_ids)
    cand_i = unique_ids // num_nodes
    cand_j = unique_ids % num_nodes
    cand_edges = np.stack([cand_i, cand_j], axis=1)

    bond_lookup = np.zeros(num_nodes * num_nodes, dtype=np.int64)
    bond_lookup[exist_i * num_nodes + exist_j] = exist_bt
    cand_bond_types = bond_lookup[unique_ids]

    return cand_edges, cand_bond_types

def build_candidate_edges_batched(
    batch, rc_probs, rc_threshold=0.3, training=True, ss_prob=0.0, ss_rc_threshold=0.5,
    local_frag_id=None, soft_rc_threshold=None,
):
    """
    批量构建候选边 — CPU 版本

    所有逐分子操作在 CPU numpy 上完成（30×30 矩阵 CPU 操作 ~100ns，
    GPU kernel launch ~10μs，CPU 快 100x），最后一次性搬回 GPU。

    v10 新增:
        local_frag_id: [N_total] 每个原子在其反应内的 fragment 局部 ID
        soft_rc_threshold: float, 跨分子 RC×RC pair 的软阈值（仅推理 / SS 使用）
    """
    device = batch.x.device
    batch_idx = batch.batch
    rc_all_pairs = CONFIG.get("rc_all_pairs", False)
    rc_min_top_k = CONFIG.get("rc_min_top_k", 0)
    max_rc_atoms = CONFIG.get("max_rc_atoms", 0)
    max_cand_edges = CONFIG.get("max_cand_edges_per_mol", 0)
    gt_drop_prob = CONFIG.get("gt_drop_prob", 0.0)

    batch_idx_cpu = batch_idx.cpu().numpy()
    edge_index_cpu = batch.edge_index.cpu().numpy()
    edge_attr_cpu = batch.edge_attr.cpu().numpy()
    rc_probs_cpu = rc_probs.detach().float().cpu().numpy()
    local_frag_id_cpu = (
        local_frag_id.cpu().numpy() if local_frag_id is not None else None
    )

    batch_size = int(batch_idx_cpu.max()) + 1
    node_counts = np.bincount(batch_idx_cpu, minlength=batch_size)
    node_offsets = np.zeros(batch_size, dtype=np.int64)
    node_offsets[1:] = np.cumsum(node_counts[:-1])

    edge_mol = batch_idx_cpu[edge_index_cpu[0]]

    y_delta_list = (
        batch.y_delta_list if hasattr(batch, "y_delta_list") else [None] * batch_size
    )

    y_delta_np_list = [
        yd.numpy() if yd is not None and yd.device.type == "cpu" else
        (yd.cpu().numpy() if yd is not None else None)
        for yd in y_delta_list
    ]

    all_cand_edges = []
    all_cand_bond_types = []
    all_edit_labels = []

    for mol_idx in range(batch_size):
        start = int(node_offsets[mol_idx])
        n = int(node_counts[mol_idx])

        emask = edge_mol == mol_idx
        mol_ei = edge_index_cpu[:, emask] - start
        mol_ea = edge_attr_cpu[emask]

        mol_rc_p = rc_probs_cpu[start : start + n]
        y_delta_np = y_delta_np_list[mol_idx]

        use_pred_rc = False
        inject_gt = False

        if training and y_delta_np is not None:
            if ss_prob > 0 and random.random() < ss_prob:
                use_pred_rc = True
                inject_gt = False
            else:
                use_pred_rc = False
                inject_gt = random.random() >= gt_drop_prob

        if use_pred_rc or not training:
            thr = ss_rc_threshold if (training and use_pred_rc) else rc_threshold
            rc_mask_np = mol_rc_p > thr
            if rc_min_top_k > 0 and rc_mask_np.sum() < rc_min_top_k:
                k = min(rc_min_top_k, n)
                topk_idx = np.argsort(-mol_rc_p)[:k]
                rc_mask_np = np.zeros(n, dtype=np.bool_)
                rc_mask_np[topk_idx] = True
        else:

            row_active = (y_delta_np != 0).any(axis=1)
            col_active = (y_delta_np != 0).any(axis=0)
            rc_mask_np = row_active | col_active

        if max_rc_atoms > 0 and rc_mask_np.sum() > max_rc_atoms:
            topk_idx = np.argsort(-mol_rc_p)[:max_rc_atoms]
            rc_mask_np = np.zeros(n, dtype=np.bool_)
            rc_mask_np[topk_idx] = True

        rc_mask_soft_np = None
        mol_local_frag_np = None
        if (
            (use_pred_rc or not training)
            and soft_rc_threshold is not None
            and local_frag_id_cpu is not None
        ):
            rc_mask_soft_np = mol_rc_p > soft_rc_threshold
            mol_local_frag_np = local_frag_id_cpu[start : start + n]

        cand_edges_np, cand_bt_np = build_candidate_edges_cpu(
            mol_ei, mol_ea, n, rc_mask_np,
            y_delta_np if inject_gt else None,
            rc_all_pairs=rc_all_pairs,
            rc_mask_soft_np=rc_mask_soft_np,
            atom_local_frag_id_np=mol_local_frag_np,
        )

        if max_cand_edges > 0 and len(cand_edges_np) > max_cand_edges:
            if y_delta_np is not None and len(cand_edges_np) > 0:
                gt_labels = y_delta_np[cand_edges_np[:, 0], cand_edges_np[:, 1]]
                gt_mask = gt_labels != 0
                gt_count = gt_mask.sum()
                if gt_count < max_cand_edges:
                    non_gt = np.where(~gt_mask)[0]
                    keep_non_gt = max_cand_edges - gt_count
                    perm = np.random.permutation(len(non_gt))[:keep_non_gt]
                    keep = np.sort(np.concatenate([np.where(gt_mask)[0], non_gt[perm]]))
                    cand_edges_np = cand_edges_np[keep]
                    cand_bt_np = cand_bt_np[keep]
            else:
                perm = np.sort(np.random.permutation(len(cand_edges_np))[:max_cand_edges])
                cand_edges_np = cand_edges_np[perm]
                cand_bt_np = cand_bt_np[perm]

        if y_delta_np is not None and len(cand_edges_np) > 0:
            labels_np = y_delta_np[cand_edges_np[:, 0], cand_edges_np[:, 1]]
        else:
            labels_np = np.zeros(len(cand_edges_np), dtype=np.int64)

        all_cand_edges.append(cand_edges_np)
        all_cand_bond_types.append(cand_bt_np)
        all_edit_labels.append(labels_np)

    all_cand_edges_gpu = [
        torch.from_numpy(e).to(device) for e in all_cand_edges
    ]
    all_cand_bond_types_gpu = [
        torch.from_numpy(b).to(device) for b in all_cand_bond_types
    ]
    all_edit_labels_gpu = [
        torch.from_numpy(l).to(device) for l in all_edit_labels
    ]

    return all_cand_edges_gpu, all_cand_bond_types_gpu, all_edit_labels_gpu

class RCSetRefiner(nn.Module):
    """v21 §44.2: RC Cross-Atom Set Refiner.

    NOTE: 命名 RCSetRefiner 而非 RCRefiner, 避开与 v20.0 的 RCRefiner (line ~3493,
    两轮 RC 精修, 用 EdgeTransformer 反馈) 冲突。两者完全不同模块, 互不影响。

    在 RC head 输出的 atom-wise raw logit 之后，接一个 mol 级 self-attention
    transformer encoder，让每个原子的 RC 概率显式条件于其他原子的 raw logit
    与 h_atom — 解决 atom-wise 独立 BCE 在 K=4/5+ 的几何衰减。

    输入:
        h_atom: [N, hidden_dim] 原子表征 (post adapter_rc, post FiLM)
        rc_logit_raw: [N] RC head 的原始 logit
        batch_idx: [N] 每个原子的 mol 编号
    输出:
        delta_logit: [N] 残差形式的修正量, 初始 ≈ 0 不破坏 warmstart
    """
    def __init__(self, hidden_dim=512, num_layers=2, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(hidden_dim + 1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.delta_head = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, h_atom, rc_logit_raw, batch_idx):
        token = torch.cat([h_atom, rc_logit_raw.unsqueeze(-1)], dim=-1)
        token = self.input_proj(token)
        token_dense, mask = to_dense_batch(token, batch_idx)
        token_refined = self.encoder(token_dense, src_key_padding_mask=~mask)
        token_refined = token_refined[mask]
        delta = self.delta_head(token_refined).squeeze(-1)
        return delta

class EdgeFeatureBuilder(nn.Module):
    """为每条候选边构建特征向量

    v10 改动: HOMO/LUMO 现在是 per-fragment 的（来自 EditGNNModel 的 global_frag_id 池化），
    所以候选边的两个端点可能来自不同分子片段。我们用 atom_frag_id 直接索引每个端点
    的 HOMO/LUMO，并构造双向 FMO gap (Nu_i HOMO - E_j LUMO 和反向)。
    物理特征维度: 13 → 17 (新增 4 个端点轨道能 + 净增 1 个反向 FMO gap)
    实际新维度: elec(1)+q_diff(1)+f_prod(4)+topo(3)+homo_i(1)+lumo_i(1)+homo_j(1)+lumo_j(1)+gap_FMO_ij(1)+gap_FMO_ji(1)+orbital_min(4) = 19
    """

    PHYSICS_DIM = 19

    def __init__(self, hidden_dim=512, use_interaction_terms=True):
        super().__init__()

        self.use_interaction_terms = use_interaction_terms
        self.bond_type_emb = nn.Embedding(11, 64)
        self.hop_dist_emb = nn.Embedding(8, 32)

        n_h = 4 if use_interaction_terms else 2
        input_dim = hidden_dim * n_h + 64 + 32 + self.PHYSICS_DIM + 1 + hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        h_nodes,
        fukui,
        charge,
        homo,
        lumo,
        cand_edges,
        cand_bond_types,
        hop_dists,
        v_global,
        edge_mol_ids=None,
        cross_mol=None,
        atom_frag_id=None,
    ):
        """
        批量模式 (推荐): 所有分子的边拼接在一起, 用全局节点索引
                h_nodes: [N_total, 512]
                fukui: [N_total, 2]
                charge: [N_total, 1]
                homo: [num_global_frags, 1], lumo: [num_global_frags, 1]  ← v10: per-fragment
                cand_edges: [K_total, 2] 全局节点索引
                cand_bond_types: [K_total]
                hop_dists: [K_total]
                v_global: [B_reactions, 512]
                edge_mol_ids: [K_total] 每条边所属反应 ID (用于索引 v_global)
                atom_frag_id: [N_total] 每个原子的全局 fragment ID (用于按端点取 HOMO/LUMO)

        Returns: [K_total, 512]
        """
        if cand_edges.size(0) == 0:
            return torch.empty((0, 512), device=h_nodes.device)

        i_idx = cand_edges[:, 0]
        j_idx = cand_edges[:, 1]

        h_i = h_nodes[i_idx]
        h_j = h_nodes[j_idx]

        if self.use_interaction_terms:
            h_prod = h_i * h_j
            h_diff = h_i - h_j

        bond_emb = self.bond_type_emb(cand_bond_types.clamp(0, 10))
        hop_emb = self.hop_dist_emb(hop_dists.clamp(0, 7))

        K = cand_edges.size(0)
        f_i = fukui[i_idx]
        f_j = fukui[j_idx]
        c_i = charge[i_idx]
        c_j = charge[j_idx]

        elec = c_i * c_j
        q_diff = torch.abs(c_i - c_j)
        f_prod = (f_i.unsqueeze(-1) * f_j.unsqueeze(-2)).reshape(K, 4)

        if atom_frag_id is not None:
            frag_i = atom_frag_id[i_idx]
            frag_j = atom_frag_id[j_idx]
            homo_i = homo[frag_i]
            lumo_i = lumo[frag_i]
            homo_j = homo[frag_j]
            lumo_j = lumo[frag_j]
        else:

            if edge_mol_ids is not None:
                homo_i = homo[edge_mol_ids]
                lumo_i = lumo[edge_mol_ids]
            else:
                homo_i = homo.expand(K, -1) if homo.dim() > 1 else homo.unsqueeze(0).expand(K, 1)
                lumo_i = lumo.expand(K, -1) if lumo.dim() > 1 else lumo.unsqueeze(0).expand(K, 1)
            homo_j = homo_i
            lumo_j = lumo_i

        gap_ij = F.softplus(lumo_j - homo_i) + 0.5
        gap_ji = F.softplus(lumo_i - homo_j) + 0.5
        gap_min = torch.minimum(gap_ij, gap_ji)
        orbital = f_prod / gap_min

        topo = torch.stack(
            [
                (hop_dists == 1).float(),
                (hop_dists == 2).float(),
                (hop_dists >= 3).float(),
            ],
            dim=-1,
        )

        physics = torch.cat([
            elec, q_diff, f_prod,
            homo_i, lumo_i, homo_j, lumo_j,
            gap_ij, gap_ji,
            orbital, topo,
        ], dim=-1)

        if cross_mol is not None:
            cross_mol_feat = cross_mol.float().unsqueeze(-1)
        else:
            cross_mol_feat = torch.zeros(K, 1, device=h_i.device)

        if edge_mol_ids is not None:
            v_global_expanded = v_global[edge_mol_ids]
        else:
            v_global_expanded = v_global.unsqueeze(0).expand(K, -1)

        cat_list = [h_i, h_j]
        if self.use_interaction_terms:
            cat_list.extend([h_prod, h_diff])
        cat_list.extend([bond_emb, hop_emb, physics, cross_mol_feat, v_global_expanded])
        feat = torch.cat(cat_list, dim=-1)

        return self.proj(feat)

def compute_hop_distances(edge_index, num_nodes, cand_edges, max_hop=7):
    """
    计算候选边中每对原子的最短路径距离 — 矩阵幂版本 (单分子兼容接口)

    Returns:
            hop_dists: [K] long, 最短路径距离 (分子内可达) 或 max_hop (不可达)
            cross_mol: [K] bool, True 表示这对原子在不同连通分量（跨分子）
    """
    if cand_edges.size(0) == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=cand_edges.device),
            torch.empty((0,), dtype=torch.bool, device=cand_edges.device),
        )

    device = cand_edges.device

    adj = torch.zeros(num_nodes, num_nodes, device=device)
    if edge_index.size(1) > 0:
        adj[edge_index[0], edge_index[1]] = 1.0

    dist = torch.full(
        (num_nodes, num_nodes), max_hop + 1, dtype=torch.long, device=device
    )
    dist.fill_diagonal_(0)

    reachable = adj > 0
    dist[reachable] = torch.where(
        dist[reachable] > 1, torch.ones_like(dist[reachable]), dist[reachable]
    )

    power = adj.clone()
    for hop in range(2, max_hop + 1):
        power = power @ adj
        newly = (power > 0) & (dist > hop)
        if not newly.any():
            break
        dist[newly] = hop

    hop_dists = dist[cand_edges[:, 0], cand_edges[:, 1]]

    cross_mol = hop_dists > max_hop
    hop_dists = hop_dists.clamp(max=max_hop)
    return hop_dists, cross_mol

def compute_hop_distances_batched(
    edge_index, num_nodes, cand_edges_global, max_hop=7,
    precomputed_hop_matrix=None,
):
    """
    批量计算所有候选边的 hop distance — 使用 scipy 稀疏 BFS

    整个 batch 的 block-diagonal 邻接矩阵天然隔离不同分子，
    一次 shortest_path 调用替代逐分子循环。

    只对候选边的 unique 源节点做 BFS，避免全图最短路。

    v20.1 速度优化: 若 batch 上挂了 precomputed_hop_matrix (collate 端 worker 预计算的
    完整 N×N int8 距离矩阵), 直接 lookup 跳过 scipy BFS — 主线程实测省 27% 时间。

    Args:
        edge_index: [2, E_total] 整个 batch 的边
        num_nodes: int, 总节点数
        cand_edges_global: [K_total, 2] 全局索引的候选边
        max_hop: int, 最大 hop 距离
        precomputed_hop_matrix: [N, N] numpy int8 (含 inf 已转为 max_hop+1) or None

    Returns:
        hop_dists: [K_total] long
        cross_mol: [K_total] bool
    """
    if cand_edges_global.size(0) == 0:
        device = edge_index.device if edge_index.size(1) > 0 else "cpu"
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.bool, device=device),
        )

    device = cand_edges_global.device

    if precomputed_hop_matrix is not None:
        cand_cpu = cand_edges_global.cpu()
        i_idx = cand_cpu[:, 0].numpy()
        j_idx = cand_cpu[:, 1].numpy()

        hop_np = precomputed_hop_matrix[i_idx, j_idx].astype(np.int64)
        hop_dists = torch.from_numpy(hop_np).to(device)
        cross_mol = hop_dists > max_hop
        hop_dists = hop_dists.clamp(max=max_hop)
        return hop_dists, cross_mol

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()
    ones = np.ones(len(src), dtype=np.float32)
    adj_sparse = csr_matrix((ones, (src, dst)), shape=(num_nodes, num_nodes))

    cand_cpu = cand_edges_global.cpu()
    i_idx = cand_cpu[:, 0].numpy()
    j_idx = cand_cpu[:, 1].numpy()
    unique_sources = np.unique(i_idx)

    dist_matrix = shortest_path(
        adj_sparse, method="D", directed=False, indices=unique_sources,
        unweighted=True
    )

    source_to_row = np.empty(num_nodes, dtype=np.int64)
    source_to_row[unique_sources] = np.arange(len(unique_sources))

    row_indices = source_to_row[i_idx]
    hop_np = dist_matrix[row_indices, j_idx]

    hop_np = np.where(np.isinf(hop_np), max_hop + 1, hop_np).astype(np.int64)

    hop_dists = torch.from_numpy(hop_np).to(device)
    cross_mol = hop_dists > max_hop
    hop_dists = hop_dists.clamp(max=max_hop)

    return hop_dists, cross_mol

class EdgeTransformer(nn.Module):
    """候选边之间的自注意力 — 核心创新"""

    def __init__(self, hidden_dim=512, num_layers=3, num_heads=8, ff_dim=1024):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, edge_features, edge_batch_ids):
        """
        edge_features: [total_K, 512] 所有分子的边特征拼接
        edge_batch_ids: [total_K] 每条边属于哪个分子

        Returns: [total_K, 512]
        """
        if edge_features.size(0) == 0:
            return edge_features

        device = edge_features.device
        D = edge_features.size(1)
        batch_size = edge_batch_ids.max().item() + 1
        counts = torch.bincount(edge_batch_ids, minlength=batch_size)
        max_k = counts.max().item()

        if max_k == 0:
            return edge_features

        offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
        offsets[1:] = counts[:-1].cumsum(0)

        local_pos = (
            torch.arange(edge_features.size(0), device=device) - offsets[edge_batch_ids]
        )

        padded = torch.zeros(batch_size, max_k, D, device=device)
        padded[edge_batch_ids, local_pos] = edge_features

        mask = torch.ones(batch_size, max_k, dtype=torch.bool, device=device)
        mask[edge_batch_ids, local_pos] = False

        use_ckpt = CONFIG.get("edge_transformer_use_checkpoint", True) if self.training else False
        if use_ckpt:
            for layer in self.transformer.layers:
                padded = torch.utils.checkpoint.checkpoint(
                    layer,
                    padded,
                    None,
                    mask,
                    False,
                    use_reentrant=False,
                )
            out = (
                self.transformer.norm(padded)
                if self.transformer.norm is not None
                else padded
            )
        else:
            out = self.transformer(padded, src_key_padding_mask=mask)

        return out[edge_batch_ids, local_pos]

class EditClassifier(nn.Module):
    """7类分类头: 0/+1/-1/+2/-2/+3/-3"""

    def __init__(self, hidden_dim=512, num_classes=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        with torch.no_grad():
            self.mlp[-1].bias[0] = 1.5
            self.mlp[-1].bias[1:] = -0.25

    def forward(self, x):
        return self.mlp(x)

class ContrastiveRCLoss(nn.Module):
    """方案三: 分子内 RC vs non-RC 对比学习

    核心思路: BCE loss 独立处理每个原子，无法利用"同一分子内 RC 原子应该彼此相似、
    与非 RC 原子应该不同"的结构先验。对比学习显式拉开 RC 和 non-RC 的表征距离，
    帮助消除化学环境相似但不参与反应的假阳性原子。

    使用 SupCon (Supervised Contrastive) 变体:
    - 正样本对: 同一分子内的 RC 原子之间
    - 负样本对: RC 原子 vs 同一分子内的 non-RC 原子
    - 温度参数控制聚焦程度: 越小越关注 hard negatives
    """

    def __init__(self, temperature=0.07, hidden_dim=512):
        super().__init__()
        self.temp = temperature

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
        )

    def forward(self, h_rc, rc_target, batch_idx):
        """
        h_rc: [N_total, 512] RC adapter 输出的原子表征
        rc_target: [N_total] float, RC 标签 (0 或 1)
        batch_idx: [N_total] 每个原子所属反应 ID

        Returns: scalar loss
        """
        if h_rc.size(0) == 0:
            return torch.tensor(0.0, device=h_rc.device)

        z = F.normalize(self.projector(h_rc), dim=-1)

        batch_size = batch_idx.max().item() + 1
        total_loss = 0.0
        valid_mols = 0

        for mol_idx in range(batch_size):
            mol_mask = batch_idx == mol_idx
            z_mol = z[mol_mask]
            rc_mol = rc_target[mol_mask]
            n = z_mol.size(0)

            pos_mask = rc_mol > 0.5
            neg_mask = rc_mol < 0.5
            n_pos = pos_mask.sum().item()
            n_neg = neg_mask.sum().item()

            if n_pos < 1 or n_neg < 1:
                continue

            sim = z_mol @ z_mol.T / self.temp

            anchor_mask = pos_mask
            anchor_indices = anchor_mask.nonzero(as_tuple=True)[0]

            if anchor_indices.size(0) < 2:

                continue

            diag_mask = torch.eye(n, dtype=torch.bool, device=z_mol.device)

            pos_pair_mask = pos_mask.unsqueeze(0) & pos_mask.unsqueeze(1) & ~diag_mask

            sim_max, _ = sim.max(dim=1, keepdim=True)
            sim_stable = sim - sim_max.detach()

            exp_sim = torch.exp(sim_stable) * (~diag_mask).float()
            log_sum_all = torch.log(exp_sim.sum(dim=1) + 1e-8)

            exp_pos = torch.exp(sim_stable) * pos_pair_mask.float()
            log_sum_pos = torch.log(exp_pos.sum(dim=1) + 1e-8)

            anchor_loss = log_sum_all[anchor_indices] - log_sum_pos[anchor_indices]
            total_loss = total_loss + anchor_loss.mean()
            valid_mols += 1

        if valid_mols == 0:
            return torch.tensor(0.0, device=h_rc.device)

        return total_loss / valid_mols

class RCRefiner(nn.Module):
    """方案一: RC 迭代精修 — 利用 EdgeTransformer 输出反馈修正 RC 预测

    核心思路: 如果一个原子的所有候选边的 edit 预测都是 class 0 (不变),
    说明 EdgeTransformer 认为它不参与反应 → 应降低其 RC 概率。
    反之，如果某个非 RC 原子的候选边被预测为非零编辑 → 应提升其 RC 概率。

    实现: 将 edge 特征按原子聚合，与原始 RC 特征门控融合，输出精修后的 RC logits。

    信息流:
    Round 1 RC logits → 候选边 → EdgeTransformer → edit 特征/置信度
                                                                                                             ↓
    Round 2:   edit 信息聚合到原子 → 门控融合原始 RC 特征 → 精修 RC logits
                                                                                                             ↓
                      重建候选边 → EdgeTransformer(Round 2) → 最终 edit 预测
    """

    def __init__(self, hidden_dim=512):
        super().__init__()

        self.edge_to_atom = nn.Sequential(
            nn.Linear(hidden_dim + 7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.rc_refine_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, h_rc, edge_features, edit_logits, cand_edges_global, num_atoms):
        """
        h_rc: [N_total, 512] 原始 RC 原子表征
        edge_features: [K_total, 512] EdgeTransformer 输出的边特征
        edit_logits: [K_total, 7] 第一轮 edit 预测的 logits
        cand_edges_global: [K_total, 2] 全局索引的候选边
        num_atoms: N_total

        Returns: refined_rc_logits [N_total]
        """
        device = h_rc.device

        if edge_features.size(0) == 0:

            return torch.zeros(num_atoms, device=device)

        edit_probs = F.softmax(edit_logits.float(), dim=-1)
        edge_msg_input = torch.cat(
            [edge_features.float(), edit_probs], dim=-1
        )
        edge_msg = self.edge_to_atom(edge_msg_input)

        i_idx = cand_edges_global[:, 0]
        j_idx = cand_edges_global[:, 1]

        all_atom_idx = torch.cat([i_idx, j_idx], dim=0)
        all_edge_msg = torch.cat([edge_msg, edge_msg], dim=0)

        all_edge_msg = all_edge_msg.float()
        atom_msg = torch.zeros(num_atoms, edge_msg.size(1), device=device)
        atom_count = torch.zeros(num_atoms, 1, device=device)
        atom_msg.scatter_add_(
            0, all_atom_idx.unsqueeze(1).expand_as(all_edge_msg), all_edge_msg
        )
        atom_count.scatter_add_(
            0,
            all_atom_idx.unsqueeze(1),
            torch.ones_like(all_atom_idx.unsqueeze(1).float()),
        )
        atom_msg = atom_msg / (atom_count + 1e-8)

        gate = self.gate(torch.cat([h_rc.float(), atom_msg], dim=-1))

        h_fused = torch.cat([h_rc.float(), atom_msg * gate], dim=-1)
        refined_logits = self.rc_refine_head(h_fused).squeeze(-1)

        return refined_logits

class KPredHead(nn.Module):
    def __init__(self, hidden_dim=512, proj_hidden=256, num_classes=8, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_hidden),
            nn.GELU(),
            nn.Linear(proj_hidden, num_classes),
        )

        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h_atom, batch_idx, v_global):
        """
        Args:
            h_atom: [N_total, hidden] atom embeddings across batch
            batch_idx: [N_total] which mol each atom belongs to
            v_global: [B, hidden] reaction-level context
        Returns:
            k_logits: [B, num_classes]
        """

        B = v_global.size(0)
        device = h_atom.device
        dtype = h_atom.dtype
        pool_sum = torch.zeros(B, h_atom.size(-1), device=device, dtype=dtype)
        pool_cnt = torch.zeros(B, 1, device=device, dtype=dtype)
        pool_sum.index_add_(0, batch_idx, h_atom)
        ones = torch.ones(h_atom.size(0), 1, device=device, dtype=dtype)
        pool_cnt.index_add_(0, batch_idx, ones)
        pool_avg = pool_sum / pool_cnt.clamp(min=1.0)
        feat = torch.cat([pool_avg, v_global], dim=-1)
        return self.proj(feat)

class EditGNNModel(nn.Module):
    def __init__(self, pretrained_model, config):
        super().__init__()
        hidden_dim = config["hidden_dim"]

        self.atom_model = pretrained_model
        for param in self.atom_model.parameters():
            param.requires_grad = False

        unfreeze_n = config.get("unfreeze_encoder_top_n", 0)
        if unfreeze_n > 0:
            encoder = self.atom_model.encoder
            num_layers = len(encoder.convs)
            for i in range(num_layers - unfreeze_n, num_layers):
                for param in encoder.convs[i].parameters():
                    param.requires_grad = True
                for param in encoder.norms[i].parameters():
                    param.requires_grad = True

        self.adapter_rc = DeepAdapterLayer(
            hidden_dim, bottleneck_dim=384,
            dropout=config.get("adapter_rc_dropout", 0.15),
        )
        self.adapter_edit = DeepAdapterLayer(
            hidden_dim, bottleneck_dim=384,
            dropout=config.get("adapter_edit_dropout", 0.1),
        )

        self.edit_inter_mol_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.edit_inter_mol_norm = nn.LayerNorm(hidden_dim)

        self.frag_descriptor = FragmentDescriptor(hidden_dim)

        cross_layers = config.get("cross_frag_layers", 2)
        cross_heads = config.get("cross_frag_heads", 8)
        self.cross_frag_attn = CrossFragmentAttention(
            hidden_dim, num_heads=cross_heads, num_layers=cross_layers
        )

        if cross_layers > 2:
            for layer_idx in range(2, cross_layers):

                attn = self.cross_frag_attn.attn_layers[layer_idx]
                nn.init.zeros_(attn.out_proj.weight)
                nn.init.zeros_(attn.out_proj.bias)

                ffn = self.cross_frag_attn.ffn_layers[layer_idx]

                nn.init.zeros_(ffn[3].weight)
                nn.init.zeros_(ffn[3].bias)

        self.rc_head = RCHead(hidden_dim)

        self.k_pred_head = None

        if config.get("use_rc_set_pred_head", True):
            self.rc_set_pred_head = RCSetPredHead(
                hidden_dim=hidden_dim,
                K_max=config.get("rc_set_pred_K_max", 10),
                num_layers=config.get("rc_set_pred_num_layers", 3),
                num_heads=config.get("rc_set_pred_num_heads", 8),
                ffn_dim=config.get("rc_set_pred_ffn_dim", 2048),
                dropout=config.get("rc_set_pred_dropout", 0.1),
                N_max=config.get("rc_set_pred_N_max", 60),
            )
        else:
            self.rc_set_pred_head = None

        if config.get("use_rc_refiner", False):
            self.rc_set_refiner = RCSetRefiner(
                hidden_dim=hidden_dim,
                num_layers=config.get("rc_refiner_layers", 2),
                num_heads=config.get("rc_refiner_heads", 8),
                dropout=config.get("rc_refiner_dropout", 0.1),
            )
        else:
            self.rc_set_refiner = None

        if config.get("use_pair_cons_gate", False):
            self.pair_cons_gate = PairConsistencyGate(config)
        else:
            self.pair_cons_gate = None

        self.edge_rc_head = EdgeRCHead(
            hidden_dim,
            num_bond_types=8,
            dropout=config.get("rc_head_dropout", 0.2),
        )

        self.rc_pair_refiner = RCPairRefiner(hidden_dim)

        if config.get("use_pair_sym_head", False):
            self.pair_sym_head = PairSymmetryHead(
                d_model=hidden_dim,
                hidden=config.get("pair_head_hidden", 256),
                dropout=config.get("rc_head_dropout", 0.2),
            )
        else:
            self.pair_sym_head = None

        if config.get("use_pair_hard_head", False):
            self.pair_hard_head = PairHardHead(
                d_model=hidden_dim,
                hidden=config.get("pair_hard_hidden", 256),
                dropout=config.get("pair_hard_dropout", 0.1),
            )
        else:
            self.pair_hard_head = None

        self.env_encoder = EnvironmentEncoder(fp_dim=1024, hidden_dim=hidden_dim)
        self.text_encoder = TextEncoder(config["bert_model"], hidden_dim=hidden_dim)
        self.cond_encoder = ConditionEncoder(hidden_dim=hidden_dim)
        self.global_project = nn.Linear(hidden_dim * 3, hidden_dim)

        self.edge_feature_builder = EdgeFeatureBuilder(
            hidden_dim,
            use_interaction_terms=config.get("edge_feat_use_interaction_terms", True),
        )

        self.edge_transformer = EdgeTransformer(
            hidden_dim,
            num_layers=config.get("n_edge_transformer_layers", 3),
            num_heads=8,
            ff_dim=512,
        )

        self.large_cand_dropout_p = 0.0
        self.edit_classifier = EditClassifier(hidden_dim, config["delta_classes"])

        if config.get("use_detr_edit_head", False):
            self.detr_edit_head = DETREditHead(
                hidden_dim=config.get("detr_edit_hidden", hidden_dim),
                n_heads=config.get("detr_edit_n_heads", 8),
                n_decoder_layers=config.get("detr_edit_n_decoder_layers", 2),
                ffn_dim=config.get("detr_edit_ffn_dim", 2048),
                dropout=config.get("detr_edit_dropout", 0.1),
                n_classes=config["delta_classes"],
            )
        else:
            self.detr_edit_head = None

        self.edit_refiner = None

        self._contrastive_loss_fn = None

        self.rc_refiner = RCRefiner(hidden_dim)

        if config.get("use_rlc_film", False):
            r_dim = config.get("rlc_dim", hidden_dim)
            r_drop = config.get("rlc_dropout", 0.1)
            self.rlc_film = RLCFiLMConditioner(
                atom_dim=hidden_dim, r_dim=r_dim, dropout=r_drop
            )
        else:
            self.rlc_film = None

        if config.get("use_role_signals", False) and _extract_role_signals is not None:
            self.role_signal_proj = RoleSignalProjector(
                in_dim=config.get("role_signal_dim", 40),
                hidden_dim=hidden_dim,
                dropout=config.get("role_signal_dropout", 0.1),
            )
        else:
            self.role_signal_proj = None

        if config.get("use_pfrh", False):
            self.pfrh = PerFragRcCalibrator(
                init_T=config.get("pfrh_init_T", 3.0),
                init_b=config.get("pfrh_init_b", 0.0),
                eps=config.get("pfrh_eps", 1e-5),
                min_frag=config.get("pfrh_min_frag", 3),
            )
        else:
            self.pfrh = None

        if config.get("use_edit_rlc", False):
            r_dim = config.get("rlc_dim", hidden_dim)
            self.edit_rlc_proj = nn.Linear(r_dim, hidden_dim)
            nn.init.zeros_(self.edit_rlc_proj.weight)
            nn.init.zeros_(self.edit_rlc_proj.bias)
        else:
            self.edit_rlc_proj = None

        if config.get("use_multi_hop_rc", False):
            self.multi_hop_rc_layers = int(config.get("multi_hop_rc_layers", 3))

            self.multi_hop_rc_gate = nn.Parameter(
                torch.full((self.multi_hop_rc_layers - 1,), 0.05)
            )

            self.multi_hop_rc_proj = nn.Linear(hidden_dim, hidden_dim)

            self._multi_hop_rc_warmup_active = True
        else:
            self.multi_hop_rc_layers = 1
            self.multi_hop_rc_gate = None
            self.multi_hop_rc_proj = None
            self._multi_hop_rc_warmup_active = False

    def _multi_hop_enhance_v_global(self, v_global, h_rc, rc_probs, batch_idx, hop_idx):
        """A1 helper: rc_probs-weighted scatter pool over h_rc, gated injection into v_global.

        Args:
            v_global: [B, D]
            h_rc: [N, D]
            rc_probs: [N] sigmoid 后概率
            batch_idx: [N] 每原子所属 batch
            hop_idx: int, 0-indexed hop (用于 gate[hop_idx])
        Returns:
            v_global_enhanced: [B, D]
        """
        B = v_global.size(0)
        D = h_rc.size(-1)
        weighted_h = rc_probs.unsqueeze(-1).to(h_rc.dtype) * h_rc
        pool_sum = torch.zeros(B, D, device=h_rc.device, dtype=h_rc.dtype)
        pool_sum.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(weighted_h), weighted_h)
        norm = torch.zeros(B, 1, device=h_rc.device, dtype=rc_probs.dtype)
        norm.scatter_add_(0, batch_idx.unsqueeze(-1), rc_probs.unsqueeze(-1))
        pool_avg = pool_sum / (norm.to(h_rc.dtype) + 1e-6)
        enhanced = self.multi_hop_rc_gate[hop_idx] * self.multi_hop_rc_proj(pool_avg)
        return v_global + enhanced

    def _build_pair_candidates_for_sym_head(
        self, rc_probs, rc_pair_index_global, batch_idx, top_k=20
    ):
        """v16 V2: 为 PairSymmetryHead 构建候选 pair。

        每反应取 rc_probs top-K 原子做两两组合 (C(k,2) 对) ∪ 全部 GT RC pair。
        Target: 1 if pair in GT else 0（按 pair_key = i*N_total + j 做 isin 判定）。

        Args:
            rc_probs: [N_total] 最终 RC 概率（sigmoid 后，post-refiner）
            rc_pair_index_global: [P_gt, 2] long 或 None — batch-global GT 成键 pair (i<j)
            batch_idx: [N_total] 每原子所属反应
            top_k: 每反应取前 K 个原子做 pair 组合

        Returns:
            cand_pair_index: [P_cand, 2] long (i<j)
            pair_targets: [P_cand] float (1=GT RC pair, 0=其他)
        """
        device = rc_probs.device
        N_total = rc_probs.size(0)
        empty_idx = torch.empty((0, 2), dtype=torch.long, device=device)
        empty_tgt = torch.empty((0,), dtype=torch.float, device=device)
        if N_total < 2 or batch_idx.numel() == 0:
            return empty_idx, empty_tgt

        batch_size = int(batch_idx.max().item()) + 1
        all_pairs = []
        for b in range(batch_size):
            atom_mask = batch_idx == b
            atom_indices = atom_mask.nonzero(as_tuple=False).squeeze(-1)
            if atom_indices.numel() < 2:
                continue
            probs_b = rc_probs[atom_indices]
            k = min(top_k, atom_indices.numel())
            _, top_local = probs_b.topk(k)
            top_atoms = atom_indices[top_local]

            top_sorted, _ = top_atoms.sort()
            combs = torch.combinations(top_sorted, r=2).long()
            all_pairs.append(combs)

        if rc_pair_index_global is not None and rc_pair_index_global.numel() > 0:
            all_pairs.append(rc_pair_index_global.long().to(device))

        if not all_pairs:
            return empty_idx, empty_tgt

        cand = torch.cat(all_pairs, dim=0)

        if rc_pair_index_global is not None and rc_pair_index_global.numel() > 0:
            gt_dev = rc_pair_index_global.long().to(device)
            gt_keys = gt_dev[:, 0] * N_total + gt_dev[:, 1]
            cand_keys = cand[:, 0] * N_total + cand[:, 1]
            targets = torch.isin(cand_keys, gt_keys).float()
        else:
            targets = torch.zeros(cand.size(0), dtype=torch.float, device=device)

        return cand, targets

    def _build_pair_hard_candidates(
        self, rc_probs, rc_pair_index_global, rc_target_flat, batch_idx, top_k=15
    ):
        """v18 N1: 为 PairHardHead 构建候选 pair（硬监督）。

        每反应候选 = A ∪ B（不去重，重复对 BCE 相当于正样本轻微加权，与 PairSymHead 一致）:
          A. GT RC 原子两两组合 — 覆盖所有 GT 正 pair + GT 内部负 pair（hardest neg）
          B. Top-K predicted RC 原子两两组合（K=15）— 训练/推理分布对齐
        Target: 1 if (i, j) in rc_pair_index_global else 0（按 pair_key = i*N_total + j 向量化 isin）。

        Args:
            rc_probs: [N_total] post-refiner RC 概率（detach）
            rc_pair_index_global: [P_gt, 2] long 或 None — batch-global GT 成键 pair (i<j)
            rc_target_flat: [N_total] 0/1 — 每原子是否 GT RC（来自 batch.y_delta 已 flat）
            batch_idx: [N_total] 每原子所属反应
            top_k: B 路每反应取前 K 个 predicted RC 原子

        Returns:
            cand_pair_index: [P_cand, 2] long (i<j)
            pair_targets: [P_cand] float (1=GT RC pair, 0=其他)
        """
        device = rc_probs.device
        N_total = rc_probs.size(0)
        empty_idx = torch.empty((0, 2), dtype=torch.long, device=device)
        empty_tgt = torch.empty((0,), dtype=torch.float, device=device)
        if N_total < 2 or batch_idx.numel() == 0:
            return empty_idx, empty_tgt

        batch_size = int(batch_idx.max().item()) + 1
        all_pairs = []
        for b in range(batch_size):
            atom_mask = batch_idx == b
            atom_indices = atom_mask.nonzero(as_tuple=False).squeeze(-1)
            if atom_indices.numel() < 2:
                continue

            if rc_target_flat is not None:
                gt_mask_b = rc_target_flat[atom_indices].bool()
                gt_atoms_b = atom_indices[gt_mask_b]
                if gt_atoms_b.numel() >= 2:
                    gt_sorted, _ = gt_atoms_b.sort()
                    a_combs = torch.combinations(gt_sorted, r=2).long()
                    all_pairs.append(a_combs)

            probs_b = rc_probs[atom_indices]
            k = min(top_k, atom_indices.numel())
            _, top_local = probs_b.topk(k)
            top_atoms = atom_indices[top_local]
            top_sorted, _ = top_atoms.sort()
            b_combs = torch.combinations(top_sorted, r=2).long()
            all_pairs.append(b_combs)

        if not all_pairs:
            return empty_idx, empty_tgt
        cand = torch.cat(all_pairs, dim=0)

        if rc_pair_index_global is not None and rc_pair_index_global.numel() > 0:
            gt_dev = rc_pair_index_global.long().to(device)
            gt_keys = gt_dev[:, 0] * N_total + gt_dev[:, 1]
            cand_keys = cand[:, 0] * N_total + cand[:, 1]
            targets = torch.isin(cand_keys, gt_keys).float()
        else:
            targets = torch.zeros(cand.size(0), dtype=torch.float, device=device)

        return cand, targets

    def _edit_inter_mol_refresh(self, h, batch_idx):
        """v10: Edit 路径专属跨分子刷新 — 单层 MHA + residual + LN

        与 RCHead 内部的 _inter_mol_cross_attn 思路一致：在 adapter_edit 之后
        再让同一反应内不同分子的原子交换一次信息，使候选边端点带有最新的跨分子上下文。
        """
        if h.size(0) == 0:
            return h
        h_padded, mask = to_dense_batch(h, batch_idx)
        B, max_N, D = h_padded.shape

        mem_estimate = B * max_N * D * 8
        if mem_estimate > 1.0e9 or max_N > 500:
            return h
        padding_mask = ~mask
        attn_out, _ = self.edit_inter_mol_attn(
            h_padded, h_padded, h_padded, key_padding_mask=padding_mask
        )
        attn_out = torch.nan_to_num(attn_out, 0.0)
        h_padded = self.edit_inter_mol_norm(h_padded + attn_out)
        return h_padded[mask]

    def _build_global_edges_and_transform(
        self,
        batch,
        h_edit,
        fukui,
        charge,
        homo,
        lumo,
        v_global,
        all_cand_edges,
        all_cand_bond_types,
        global_frag_id,
    ):
        """共享的边特征构建 + EdgeTransformer 前向逻辑 (Round 1 & Round 2 复用)

        向量化版本: 一次性拼接所有分子的候选边，批量计算 hop distance，
        消除逐分子 Python for-loop。
        """
        device = batch.x.device
        batch_idx = batch.batch
        batch_size = batch_idx.max().item() + 1
        node_counts = torch.bincount(batch_idx, minlength=batch_size)
        node_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
        node_offsets[1:] = node_counts[:-1].cumsum(0)

        global_cand_edges_list = []
        global_bond_types_list = []
        edge_mol_ids_list = []

        for mol_idx in range(batch_size):
            cand_edges = all_cand_edges[mol_idx]
            if cand_edges.size(0) == 0:
                continue
            start = node_offsets[mol_idx].item()
            global_cand_edges_list.append(cand_edges + start)
            global_bond_types_list.append(all_cand_bond_types[mol_idx])
            edge_mol_ids_list.append(
                torch.full(
                    (cand_edges.size(0),), mol_idx, dtype=torch.long, device=device
                )
            )

        if global_cand_edges_list:
            all_cand_global = torch.cat(global_cand_edges_list, dim=0)
            all_bond_types_cat = torch.cat(global_bond_types_list, dim=0)
            edge_mol_ids_cat = torch.cat(edge_mol_ids_list, dim=0)

            num_total_nodes = batch_idx.size(0)
            precomputed_hop = getattr(batch, "precomputed_hop_matrix", None)
            all_hop_dists_cat, all_cross_mol_cat = compute_hop_distances_batched(
                batch.edge_index, num_total_nodes, all_cand_global,
                precomputed_hop_matrix=precomputed_hop,
            )

            all_edge_feats = self.edge_feature_builder(
                h_edit,
                fukui,
                charge,
                homo,
                lumo,
                all_cand_global,
                all_bond_types_cat,
                all_hop_dists_cat,
                v_global,
                edge_mol_ids=edge_mol_ids_cat,
                cross_mol=all_cross_mol_cat,
                atom_frag_id=global_frag_id,
            )
            transformed = self.edge_transformer(all_edge_feats, edge_mol_ids_cat)

            lcd_p = getattr(self, "large_cand_dropout_p", 0.0)
            if self.training and lcd_p > 0 and edge_mol_ids_cat.numel() > 0:
                _lcd_thr = CONFIG.get("large_cand_dropout_threshold", 100)
                _counts_per_mol = torch.bincount(
                    edge_mol_ids_cat, minlength=batch_size
                )
                _edge_mol_cnt = _counts_per_mol[edge_mol_ids_cat]
                _large_mask = _edge_mol_cnt > _lcd_thr
                if _large_mask.any():
                    _dropped = F.dropout(
                        transformed[_large_mask], p=lcd_p, training=True
                    )
                    transformed = transformed.clone()
                    transformed[_large_mask] = _dropped

            return transformed, all_cand_global, edge_mol_ids_cat
        else:
            empty_e = torch.empty((0, CONFIG["hidden_dim"]), device=device)
            empty_idx = torch.empty((0, 2), dtype=torch.long, device=device)
            empty_mol = torch.empty((0,), dtype=torch.long, device=device)
            return empty_e, empty_idx, empty_mol

    def forward(self, batch, training=True, phase=2, ss_prob=0.0, rc_refine=0.0):
        """
        Args:
                rc_refine: RC 精修强度 (0.0=关闭, 0~1.0=warmup, 1.0=全强度)。
                           refined_logits 会乘以此系数再加到 rc_logits 上。
        """
        device = batch.x.device

        if hasattr(batch, 'precomputed_global_frag_id'):
            global_frag_id = batch.precomputed_global_frag_id.to(device)
            local_frag_id = batch.precomputed_local_frag_id.to(device)
            num_global_frags = batch.precomputed_num_global_frags

            if isinstance(num_global_frags, torch.Tensor):
                num_global_frags = num_global_frags.item()
        else:
            num_atoms = batch.x.size(0)
            global_frag_id, local_frag_id, num_global_frags = compute_frag_ids_global(
                batch.edge_index, batch.batch, num_atoms
            )

        with torch.no_grad():
            h_atom, fukui, charge, homo, lumo, graph_emb = self.atom_model(
                batch.x, batch.edge_index, batch.edge_attr, global_frag_id
            )
            h_atom = torch.nan_to_num(h_atom, 0.0)
            fukui = torch.nan_to_num(fukui, 0.0)
            fukui = torch.clamp(fukui, -10.0, 10.0)
            charge = torch.nan_to_num(charge, 0.0)
            charge = torch.clamp(charge, -10.0, 10.0)
            homo = torch.nan_to_num(homo, 0.0).clamp(-10.0, 10.0)
            lumo = torch.nan_to_num(lumo, 0.0).clamp(-10.0, 10.0)
            graph_emb = torch.nan_to_num(graph_emb, 0.0)

        frag_desc = self.frag_descriptor(
            batch.x, fukui, charge, batch.edge_index,
            global_frag_id, num_global_frags, homo, lumo, graph_emb,
        )

        atom_frag_context = frag_desc[global_frag_id]

        h_cross = self.cross_frag_attn(h_atom, batch.batch, frag_context=atom_frag_context)

        h_rc = self.adapter_rc(h_cross)

        v_env = self.env_encoder(batch.env_x_all, batch.env_batch)
        v_text = self.text_encoder(batch.input_ids, batch.attention_mask)
        v_cond = self.cond_encoder(batch.temp_id, batch.time_val, batch.ab_flags)
        v_global = self.global_project(torch.cat([v_env, v_text, v_cond], dim=-1))

        if (self.role_signal_proj is not None
                and getattr(batch, "role_signals", None) is not None):
            role_emb = self.role_signal_proj(
                batch.role_signals.to(device=v_global.device, dtype=v_global.dtype)
            )
            v_global = v_global + role_emb

        if self.rlc_film is not None:
            h_rc = self.rlc_film(h_rc, v_global, batch.batch)

        k_logits = None
        if self.k_pred_head is not None:
            k_logits = self.k_pred_head(h_rc, batch.batch, v_global)

        need_h_for_pair = (self.pair_sym_head is not None) or (self.pair_hard_head is not None)

        def _call_rc_head(v_glob):
            if need_h_for_pair:
                return self.rc_head(
                    h_rc, fukui, charge, batch.edge_index, batch.batch, v_glob,
                    frag_id=global_frag_id,
                    return_h_for_pair=True,
                    global_frag_id=global_frag_id,
                    num_global_frags=num_global_frags,
                    homo_frag=homo, lumo_frag=lumo,
                )
            else:
                logits = self.rc_head(
                    h_rc, fukui, charge, batch.edge_index, batch.batch, v_glob,
                    frag_id=global_frag_id,
                    global_frag_id=global_frag_id,
                    num_global_frags=num_global_frags,
                    homo_frag=homo, lumo_frag=lumo,
                )
                return logits, None

        rc_logits_raw, h_for_pair = _call_rc_head(v_global)

        if (self.multi_hop_rc_gate is not None
                and not self._multi_hop_rc_warmup_active
                and self.multi_hop_rc_layers > 1):
            v_global_iter = v_global
            for hop_idx in range(self.multi_hop_rc_layers - 1):
                rc_probs_iter = torch.sigmoid(rc_logits_raw).detach()
                v_global_iter = self._multi_hop_enhance_v_global(
                    v_global_iter, h_rc, rc_probs_iter, batch.batch, hop_idx
                )
                rc_logits_raw, h_for_pair = _call_rc_head(v_global_iter)

        if self.rc_set_refiner is not None:
            rc_set_refiner_delta = self.rc_set_refiner(h_rc, rc_logits_raw, batch.batch)
            rc_logits_refined = rc_logits_raw + rc_set_refiner_delta
        else:
            rc_set_refiner_delta = None
            rc_logits_refined = rc_logits_raw

        rc_logits_raw = rc_logits_refined
        rc_probs_raw = torch.sigmoid(rc_logits_raw)

        refiner_delta = self.rc_pair_refiner(
            h_rc, rc_probs_raw, batch.edge_index, batch.batch, global_frag_id,
            top_k=CONFIG.get("rc_pair_refine_top_k", 6),
        )
        rc_logits = rc_logits_raw + refiner_delta
        rc_probs = torch.sigmoid(rc_logits)

        bond_logits = self.edge_rc_head(h_rc, batch.edge_index, batch.edge_attr)

        rc_pair_index_global = None
        if hasattr(batch, "precomputed_rc_target_flat"):

            rc_target_flat = batch.precomputed_rc_target_flat.to(device)
            rc_pair_index_global = batch.precomputed_rc_pair_index.to(device)
        elif hasattr(batch, "y_delta_list"):
            rc_targets = []
            pair_chunks = []
            atom_offset = 0
            for yd in batch.y_delta_list:
                yd_dev = yd.to(device)
                row_active = (yd_dev != 0).any(dim=1)
                col_active = (yd_dev != 0).any(dim=0)
                rc_targets.append((row_active | col_active).float())

                if yd_dev.dim() == 2 and yd_dev.numel() > 0:
                    triu = torch.triu(yd_dev != 0, diagonal=1)
                    pairs_local = triu.nonzero(as_tuple=False)
                    if pairs_local.numel() > 0:
                        pair_chunks.append(pairs_local + atom_offset)
                atom_offset += yd_dev.size(0) if yd_dev.dim() == 2 else 0
            rc_target_flat = torch.cat(rc_targets, dim=0)
            if pair_chunks:
                rc_pair_index_global = torch.cat(pair_chunks, dim=0).long()
            else:
                rc_pair_index_global = torch.empty((0, 2), dtype=torch.long, device=device)
        else:
            rc_target_flat = None

        pair_cand_index = None
        pair_targets = None
        pair_logits = None
        if self.pair_sym_head is not None and h_for_pair is not None:
            pair_cand_index, pair_targets = self._build_pair_candidates_for_sym_head(
                rc_probs.detach(),
                rc_pair_index_global,
                batch.batch,
                top_k=CONFIG.get("pair_head_top_k", 20),
            )
            if pair_cand_index.size(0) > 0:
                pair_logits = self.pair_sym_head(
                    h_for_pair[pair_cand_index[:, 0]],
                    h_for_pair[pair_cand_index[:, 1]],
                )
            else:
                pair_logits = torch.empty((0,), device=device)

        pair_hard_index = None
        pair_hard_target = None
        pair_hard_logits = None
        if self.pair_hard_head is not None and h_for_pair is not None:
            pair_hard_index, pair_hard_target = self._build_pair_hard_candidates(
                rc_probs.detach(),
                rc_pair_index_global,
                rc_target_flat,
                batch.batch,
                top_k=CONFIG.get("pair_hard_top_k", 15),
            )
            if pair_hard_index.size(0) > 0:
                pair_hard_logits = self.pair_hard_head(h_for_pair, pair_hard_index)
            else:
                pair_hard_logits = torch.empty((0,), device=device)

        rc_logits_pre_gate = rc_logits
        if (self.pair_cons_gate is not None
                and pair_hard_logits is not None
                and pair_hard_index is not None
                and (self.training or CONFIG.get("pair_cons_infer", True))):
            rc_logits = self.pair_cons_gate(
                rc_logits, pair_hard_logits, pair_hard_index, n_atoms=rc_logits.size(0)
            )
            rc_probs = torch.sigmoid(rc_logits)

        rc_logits_pre_pfrh = rc_logits
        if self.pfrh is not None:
            rc_logits = self.pfrh(rc_logits, global_frag_id, num_global_frags)
            rc_probs = torch.sigmoid(rc_logits)

        rc_logit_clamp = float(CONFIG.get("rc_logit_clamp", 0.0))
        if rc_logit_clamp > 0:
            rc_logits = rc_logits.clamp(-rc_logit_clamp, rc_logit_clamp)
            rc_probs = torch.sigmoid(rc_logits)

        presence_logits = atom_logits_set = mask_dense_set = atom_idx_dense_set = None
        rc_anchor_emb_v34 = None
        if self.rc_set_pred_head is not None:

            reactant_mask_v33j = torch.ones(h_rc.size(0), dtype=torch.bool, device=device)
            if self.detr_edit_head is not None:
                presence_logits, atom_logits_set, mask_dense_set, atom_idx_dense_set, rc_anchor_emb_v34 =                    self.rc_set_pred_head(h_rc, batch.batch, reactant_mask_v33j, return_queries=True)
            else:
                presence_logits, atom_logits_set, mask_dense_set, atom_idx_dense_set =                    self.rc_set_pred_head(h_rc, batch.batch, reactant_mask_v33j)

        if phase == 1:
            return {
                "rc_logits": rc_logits,
                "rc_logits_pre_gate": rc_logits_pre_gate,
                "rc_probs": rc_probs,
                "rc_target_flat": rc_target_flat,
                "k_logits": k_logits,
                "h_rc": h_rc,
                "bond_logits": bond_logits,
                "rc_pair_index": rc_pair_index_global,

                "pair_logits": pair_logits,
                "pair_cand_index": pair_cand_index,
                "pair_targets": pair_targets,

                "pair_hard_logits": pair_hard_logits,
                "pair_hard_index": pair_hard_index,
                "pair_hard_target": pair_hard_target,
                "edit_logits": torch.empty((0, CONFIG["delta_classes"]), device=device),
                "edit_labels": torch.empty((0,), dtype=torch.long, device=device),
                "cand_edges": [],
                "cand_bond_types": [],
                "num_cand_edges": 0,
                "refined_rc_logits": None,

                "fukui": fukui,
                "global_frag_id": global_frag_id,
                "num_global_frags": num_global_frags,

                "v_global": v_global,
                "batch_idx": batch.batch,

                "all_cand_global": torch.empty(
                    (0, 2), dtype=torch.long, device=device
                ),
                "edge_mol_ids": torch.empty((0,), dtype=torch.long, device=device),

                "presence_logits": presence_logits,
                "atom_logits_set": atom_logits_set,
                "mask_dense_set": mask_dense_set,
                "atom_idx_dense_set": atom_idx_dense_set,
                "reactant_mask_v33j": (
                    torch.ones(h_rc.size(0), dtype=torch.bool, device=device)
                    if self.rc_set_pred_head is not None else None
                ),
            }

        h_edit = self.adapter_edit(h_cross)

        if self.rlc_film is not None:
            if CONFIG.get("rlc_film_edit_detach", False):
                v_per_atom_e = v_global[batch.batch].detach()
                gamma_e = self.rlc_film.mlp_gamma(v_per_atom_e)
                beta_e = self.rlc_film.mlp_beta(v_per_atom_e)
                h_edit = (1.0 + gamma_e) * h_edit + beta_e
            else:
                h_edit = self.rlc_film(h_edit, v_global, batch.batch)

        h_edit = self._edit_inter_mol_refresh(h_edit, batch.batch)

        all_cand_edges, all_cand_bond_types, all_edit_labels = (
            build_candidate_edges_batched(
                batch,
                rc_probs,
                rc_threshold=CONFIG["ss_rc_threshold"],
                training=training,
                ss_prob=ss_prob,
                ss_rc_threshold=CONFIG["ss_rc_threshold"],
                local_frag_id=local_frag_id,
                soft_rc_threshold=CONFIG.get("soft_rc_threshold", None),
            )
        )

        transformed_r1, all_cand_global_r1, edge_mol_ids_r1 = (
            self._build_global_edges_and_transform(
                batch,
                h_edit,
                fukui,
                charge,
                homo,
                lumo,
                v_global,
                all_cand_edges,
                all_cand_bond_types,
                global_frag_id,
            )
        )

        if self.edit_rlc_proj is not None and transformed_r1.size(0) > 0:
            r_per_edge_r1 = self.edit_rlc_proj(v_global[edge_mol_ids_r1])
            transformed_r1 = transformed_r1 + r_per_edge_r1

        if self.detr_edit_head is not None and rc_anchor_emb_v34 is not None:
            _B_anchors = rc_anchor_emb_v34.size(0)
            _K_max_anchors = rc_anchor_emb_v34.size(1)
            _rc_anchor_mask_r1 = torch.zeros(
                _B_anchors, _K_max_anchors, dtype=torch.bool, device=device
            )
            edit_logits_r1 = self.detr_edit_head(
                transformed_r1, edge_mol_ids_r1, rc_anchor_emb_v34, _rc_anchor_mask_r1
            )
        else:
            edit_logits_r1 = self.edit_classifier(transformed_r1)

        refined_rc_logits = None
        if rc_refine > 0 and transformed_r1.size(0) > 0:
            refined_rc_logits_raw = self.rc_refiner(
                h_rc.detach(),
                transformed_r1.detach(),
                edit_logits_r1.detach(),
                all_cand_global_r1,
                h_rc.size(0),
            )

            refined_rc_logits = refined_rc_logits_raw * rc_refine

            refined_rc_probs = torch.sigmoid(rc_logits + refined_rc_logits)

            del transformed_r1, edit_logits_r1
            torch.cuda.empty_cache()

            all_cand_edges_r2, all_cand_bond_types_r2, all_edit_labels_r2 = (
                build_candidate_edges_batched(
                    batch,
                    refined_rc_probs,
                    rc_threshold=CONFIG["ss_rc_threshold"],
                    training=training,
                    ss_prob=ss_prob,
                    ss_rc_threshold=CONFIG["ss_rc_threshold"],
                    local_frag_id=local_frag_id,
                    soft_rc_threshold=CONFIG.get("soft_rc_threshold", None),
                )
            )
            transformed_r2, _, edge_mol_ids_r2 = self._build_global_edges_and_transform(
                batch,
                h_edit,
                fukui,
                charge,
                homo,
                lumo,
                v_global,
                all_cand_edges_r2,
                all_cand_bond_types_r2,
                global_frag_id,
            )

            if self.edit_rlc_proj is not None and transformed_r2.size(0) > 0:
                r_per_edge_r2 = self.edit_rlc_proj(v_global[edge_mol_ids_r2])
                transformed_r2 = transformed_r2 + r_per_edge_r2

            if self.detr_edit_head is not None and rc_anchor_emb_v34 is not None:
                _B_anchors_r2 = rc_anchor_emb_v34.size(0)
                _K_max_anchors_r2 = rc_anchor_emb_v34.size(1)
                _rc_anchor_mask_r2 = torch.zeros(
                    _B_anchors_r2, _K_max_anchors_r2, dtype=torch.bool, device=device
                )
                edit_logits = self.detr_edit_head(
                    transformed_r2, edge_mol_ids_r2, rc_anchor_emb_v34, _rc_anchor_mask_r2
                )
            else:
                edit_logits = self.edit_classifier(transformed_r2)
            all_cand_edges = all_cand_edges_r2
            all_cand_bond_types = all_cand_bond_types_r2
            all_edit_labels = all_edit_labels_r2
            rc_probs = refined_rc_probs

        else:
            edit_logits = edit_logits_r1

        if all_edit_labels:
            all_edit_labels_cat = torch.cat(all_edit_labels, dim=0)
        else:
            all_edit_labels_cat = torch.empty((0,), dtype=torch.long, device=device)

        batch_size_v20_2 = batch.batch.max().item() + 1 if batch.batch.numel() > 0 else 0
        if batch_size_v20_2 > 0 and all_cand_edges:
            _node_counts = torch.bincount(batch.batch, minlength=batch_size_v20_2)
            _node_offsets_v20_2 = torch.cat(
                [
                    torch.zeros(1, device=device, dtype=torch.long),
                    _node_counts.cumsum(0)[:-1],
                ]
            )
            _global_list = []
            _mol_id_list = []
            for _mi, _ce in enumerate(all_cand_edges):
                if _ce.numel() == 0:
                    continue
                _global_list.append(_ce.long() + _node_offsets_v20_2[_mi])
                _mol_id_list.append(
                    torch.full(
                        (_ce.size(0),),
                        _mi,
                        dtype=torch.long,
                        device=device,
                    )
                )
            if _global_list:
                all_cand_global = torch.cat(_global_list, dim=0)
                edge_mol_ids = torch.cat(_mol_id_list, dim=0)
            else:
                all_cand_global = torch.empty(
                    (0, 2), dtype=torch.long, device=device
                )
                edge_mol_ids = torch.empty((0,), dtype=torch.long, device=device)
        else:
            all_cand_global = torch.empty((0, 2), dtype=torch.long, device=device)
            edge_mol_ids = torch.empty((0,), dtype=torch.long, device=device)

        return {
            "rc_logits": rc_logits,
            "rc_logits_pre_gate": rc_logits_pre_gate,
            "rc_probs": rc_probs,
            "rc_target_flat": rc_target_flat,
            "k_logits": k_logits,
            "h_rc": h_rc,
            "bond_logits": bond_logits,
            "rc_pair_index": rc_pair_index_global,

            "pair_logits": pair_logits,
            "pair_cand_index": pair_cand_index,
            "pair_targets": pair_targets,

            "pair_hard_logits": pair_hard_logits,
            "pair_hard_index": pair_hard_index,
            "pair_hard_target": pair_hard_target,
            "edit_logits": edit_logits,
            "edit_labels": all_edit_labels_cat,
            "cand_edges": all_cand_edges,
            "cand_bond_types": all_cand_bond_types,
            "num_cand_edges": sum(e.size(0) for e in all_cand_edges),
            "refined_rc_logits": refined_rc_logits,

            "fukui": fukui,
            "global_frag_id": global_frag_id,
            "num_global_frags": num_global_frags,

            "v_global": v_global,
            "batch_idx": batch.batch,

            "all_cand_global": all_cand_global,
            "edge_mol_ids": edge_mol_ids,

            "presence_logits": presence_logits,
            "atom_logits_set": atom_logits_set,
            "mask_dense_set": mask_dense_set,
            "atom_idx_dense_set": atom_idx_dense_set,
            "reactant_mask_v33j": (
                torch.ones(h_rc.size(0), dtype=torch.bool, device=device)
                if self.rc_set_pred_head is not None else None
            ),
        }

def _k_weight_lookup(K_value, weights_dict):
    """根据 K 值查表权重. K∈{0,2,3,4}; K≥5 用 '5+'."""
    if K_value <= 0:
        return float(weights_dict.get(0, 1.0))
    if K_value <= 4:
        return float(weights_dict.get(K_value, 1.0))
    return float(weights_dict.get("5+", weights_dict.get(5, 1.0)))

def compute_set_recall_loss(
    rc_logits, rc_target_flat, batch_idx, k_weights, p_clamp_min=1e-6, max_clamp=5.0
):
    """v20.4 §42.3 Set-Selection Recall Loss.

    对每个 mol b 的 K 个 GT_RC 原子:
        l_b = -mean_{i ∈ GT_RC_b} log( sigmoid(rc_logit_i).clamp(p_clamp_min, 1-p_clamp_min) )
        l_b = min(l_b, max_clamp)
        l_b *= w_K (K-aware lookup)

    返回: scalar = mean over all mols (K=0 mols 仍计入但 weight=0.5).
    数学性质: 全程光滑可微, p clamp 防 log(0), max_clamp 防单 mol 爆炸.
    """
    device = rc_logits.device
    rc_logits = rc_logits.float()
    p = torch.sigmoid(rc_logits).clamp(p_clamp_min, 1.0 - p_clamp_min)
    pos_mask = rc_target_flat.bool() if rc_target_flat.dtype != torch.bool else rc_target_flat

    bs = int(batch_idx.max().item()) + 1
    if bs == 0:
        return torch.tensor(0.0, device=device)

    log_p_pos = -torch.log(p)

    K_per_mol = torch.zeros(bs, device=device)
    sum_per_mol = torch.zeros(bs, device=device)
    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    if pos_indices.numel() > 0:
        K_per_mol.index_add_(0, batch_idx[pos_indices], torch.ones(pos_indices.numel(), device=device))
        sum_per_mol.index_add_(0, batch_idx[pos_indices], log_p_pos[pos_indices])

    safe_K = K_per_mol.clamp(min=1.0)
    l_per_mol = (sum_per_mol / safe_K).clamp(max=max_clamp)

    l_per_mol = torch.where(K_per_mol > 0, l_per_mol, torch.zeros_like(l_per_mol))

    K_cpu = K_per_mol.detach().cpu().long().tolist()
    w_per_mol = torch.tensor(
        [_k_weight_lookup(int(k), k_weights) for k in K_cpu],
        device=device, dtype=l_per_mol.dtype,
    )
    weighted = (l_per_mol * w_per_mol).sum() / max(bs, 1)
    return weighted

def get_w_pointer_rc_v22(epoch, base, warmup_end, decay_end):
    """v22 §45.3: Pointer Loss warmup-only 衰减.
    ep1..warmup_end: full base (RC head burn-in 期约束 logit 分布)
    warmup_end..decay_end: 线性衰减
    >= decay_end: 0 (BCE 接管)

    v21 数据显示 train_l_pointer_rc 在 ep4 后饱和, 长期训练只是噪声, 故 ep5 后衰减.
    """
    if epoch is None or epoch <= warmup_end:
        return base
    if epoch >= decay_end:
        return 0.0
    return base * (decay_end - epoch) / max(decay_end - warmup_end, 1)

def compute_mol_pointer_rc_loss(rc_logits, rc_target_flat, batch_idx):
    """v21 §44.3: Mol-Level Pointer Loss.

    对每个 mol b:
      logits_b = rc_logits[atoms in mol b]
      log_probs_b = log_softmax(logits_b)   # mol 级 softmax (强制原子之间竞争)
      l_b = -mean( log_probs_b[i] for i in GT_RC atoms in mol b )

    返回: scalar = mean over mols (跳过 K=0 mol).

    与 set_recall (-log sigmoid(logit)) 的本质区别:
      - sigmoid 是独立 per-atom, partition function 是 1+e^(-x), 与其他原子无关
        → 梯度 ∂L/∂x_i 只依赖 x_i, 与 BCE 同方向
      - softmax 的 partition function = sum_j exp(x_j) over atoms in mol
        → 梯度 ∂L/∂x_i 依赖所有 x_j, 强制原子之间竞争
        → 拉高 GT atom logit 的同时压低非 GT atom logit
    """
    device = rc_logits.device
    rc_logits = rc_logits.float()
    pos_mask = rc_target_flat.bool() if rc_target_flat.dtype != torch.bool else rc_target_flat

    bs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
    if bs == 0:
        return torch.tensor(0.0, device=device)

    losses = []
    for m in range(bs):
        mol_mask = (batch_idx == m)
        if mol_mask.sum() == 0:
            continue
        target_m = pos_mask[mol_mask]
        if target_m.sum() == 0:
            continue
        logits_m = rc_logits[mol_mask]
        log_probs = torch.log_softmax(logits_m, dim=-1)
        gt_log_probs = log_probs[target_m]
        losses.append(-gt_log_probs.mean())

    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()

def compute_set_precision_loss(
    rc_logits, rc_target_flat, batch_idx, k_weights, min_K=3,
    p_clamp_min=1e-6, max_clamp=5.0,
):
    """v20.4 §42.3 Set-Selection Precision Loss (K≥min_K only).

    对每个 K≥min_K 的 mol, 取负样本 atoms (非 GT_RC):
        l_b = -mean_{i ∉ GT_RC_b} log( 1 - sigmoid(rc_logit_i).clamp(...) )
        l_b *= w_K

    K<min_K 的 mol 跳过 (§42.3.2: K=2 已 92.4%, FP 防护边际效用极小).
    返回: scalar = mean over qualifying mols.
    """
    device = rc_logits.device
    rc_logits = rc_logits.float()
    p = torch.sigmoid(rc_logits).clamp(p_clamp_min, 1.0 - p_clamp_min)
    pos_mask = rc_target_flat.bool() if rc_target_flat.dtype != torch.bool else rc_target_flat
    neg_mask = ~pos_mask

    bs = int(batch_idx.max().item()) + 1
    if bs == 0:
        return torch.tensor(0.0, device=device)

    log_q_neg = -torch.log(1.0 - p)
    K_per_mol = torch.zeros(bs, device=device)
    neg_count_per_mol = torch.zeros(bs, device=device)
    sum_neg_per_mol = torch.zeros(bs, device=device)

    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
    if pos_indices.numel() > 0:
        K_per_mol.index_add_(0, batch_idx[pos_indices], torch.ones(pos_indices.numel(), device=device))
    neg_indices = neg_mask.nonzero(as_tuple=True)[0]
    if neg_indices.numel() > 0:
        neg_count_per_mol.index_add_(0, batch_idx[neg_indices], torch.ones(neg_indices.numel(), device=device))
        sum_neg_per_mol.index_add_(0, batch_idx[neg_indices], log_q_neg[neg_indices])

    safe_neg = neg_count_per_mol.clamp(min=1.0)
    l_per_mol = (sum_neg_per_mol / safe_neg).clamp(max=max_clamp)

    qualify_mask = K_per_mol >= min_K
    if not qualify_mask.any():
        return torch.tensor(0.0, device=device)

    K_cpu = K_per_mol.detach().cpu().long().tolist()
    w_per_mol = torch.tensor(
        [_k_weight_lookup(int(k), k_weights) for k in K_cpu],
        device=device, dtype=l_per_mol.dtype,
    )
    qualify_count = qualify_mask.sum().clamp(min=1).float()
    weighted = (l_per_mol * w_per_mol * qualify_mask.float()).sum() / qualify_count
    return weighted

@torch.no_grad()
def compute_main_product_atoms(reactant_edge_index, edge_attr, y_delta_matrix, n_atoms):
    """计算主产物原子掩码 (CPU, 单 mol).

    步骤:
      1. 反应物图 (edge_index, edge_attr) 提取键级 dict {(i,j): order}, i<j
      2. y_delta 7-class → CLASS_TO_DELTA → 每个 (i,j) 加上键级变化
      3. 产物边 = (r_order + delta) > 0.5 的所有 (i,j) (含芳香键 1.5 阈值)
      4. Union-Find 求最大连通分量 = 主产物原子集

    返回: torch.bool tensor [n_atoms], True 为主产物原子.
    """
    if n_atoms == 0:
        return torch.zeros(0, dtype=torch.bool)

    src = reactant_edge_index[0].tolist()
    dst = reactant_edge_index[1].tolist()
    orders = edge_attr.tolist() if edge_attr.numel() > 0 else []
    bond_dict = {}
    for k in range(len(src)):
        i, j = src[k], dst[k]
        if i < j:
            bond_dict[(i, j)] = float(orders[k])

    yd = y_delta_matrix
    if yd.dim() != 2:
        return torch.ones(n_atoms, dtype=torch.bool)
    nz = (yd != 0).triu(diagonal=1).nonzero(as_tuple=False).tolist()
    pair_set = set(bond_dict.keys())
    for ij in nz:
        pair_set.add((int(ij[0]), int(ij[1])))

    deltas = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j) in pair_set:
        cls = int(yd[i, j].item())
        if cls < 0 or cls >= 7:
            cls = 0
        d = deltas[cls]
        r_order = bond_dict.get((i, j), 0.0)
        p_order = r_order + d
        if p_order > 0.5:
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj

    sizes = {}
    for x in range(n_atoms):
        r = find(x)
        sizes[r] = sizes.get(r, 0) + 1
    if not sizes:
        return torch.zeros(n_atoms, dtype=torch.bool)
    main_root = max(sizes, key=lambda k: sizes[k])
    out = torch.zeros(n_atoms, dtype=torch.bool)
    for x in range(n_atoms):
        if find(x) == main_root:
            out[x] = True
    return out

def compute_all_atom_hit_loss(rc_logits, rc_target_flat, batch_idx):
    """v18 N2: per-reaction all-atom-hit loss — 攻根本原因 5（mol_acc 数学错配）。

    L = mean_b ( mean_{i ∈ GT_RC_b} [ -logσ(rc_logit_i) ] )

    - 外层按反应 b 均值，消除"大分子 GT RC 数量多 → sum 梯度偏大"的偏置（v15 count_penalty 失败教训）
    - 内层 -logsigmoid 对低 prob 原子自动放大梯度（hard RC @ prob≈0.2 的那种）
    - 与 Focal γ=1.5 区分：Focal 放大 atom-level hard，本 loss 放大 reaction-level 最弱 RC
    - 纯 torch 实现 scatter_mean，避免 torch_scatter 依赖
    """
    device = rc_logits.device
    rc_logits = rc_logits.float()
    pos_mask = rc_target_flat.bool() if rc_target_flat.dtype != torch.bool else rc_target_flat
    if not pos_mask.any():
        return torch.tensor(0.0, device=device)
    log_probs = F.logsigmoid(rc_logits[pos_mask])
    pos_batch_idx = batch_idx[pos_mask].long()
    uniq, inv = torch.unique(pos_batch_idx, return_inverse=True)
    B_prime = uniq.numel()
    sum_per = torch.zeros(B_prime, device=device, dtype=log_probs.dtype)
    cnt_per = torch.zeros(B_prime, device=device, dtype=log_probs.dtype)
    sum_per.index_add_(0, inv, log_probs)
    cnt_per.index_add_(0, inv, torch.ones_like(log_probs))
    per_reaction_loss = -sum_per / cnt_per.clamp_min(1.0)
    return per_reaction_loss.mean()

def focal_loss(logits, targets, gamma=2.0, alpha=None, label_smoothing=0.0, reduction="mean"):
    """
    多分类 Focal Loss: FL(p_t) = -α_t (1-p_t)^γ log(p_t)

    解决 CE + class_weights 在极端不平衡下的 mode collapse:
    - CE + 3x weight: 对 class 0 施加恒定 3x 梯度偏差 → 一旦分类头 bias 偏移就不可逆
    - Focal Loss: γ 动态降低 well-classified 样本的梯度 → 不会产生系统性偏差

    Args:
            logits: [N, C] 未归一化的 logits
            targets: [N] 类别标签
            gamma: 聚焦参数，越大越抑制 easy examples（默认 2.0）
            alpha: [C] 类别权重（可选），用于轻度类别再平衡
            label_smoothing: 标签平滑系数
            reduction: "mean" | "none" — v20.2 加 "none" 用于 motif 加权
    """

    logits = logits.float()
    ce = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing
    )
    p_t = torch.exp(-ce)
    focal_weight = (1.0 - p_t) ** gamma

    if alpha is not None:
        alpha_t = alpha[targets]
        focal_weight = alpha_t * focal_weight

    per_sample = focal_weight * ce
    if reduction == "none":
        return per_sample
    return per_sample.mean()

def apply_motif_hard_neg_weights(
    edit_logits, edit_labels, all_cand_global, atom_z, rules
):
    """v20.2: 基于 (target_cls, pred_cls, i_elem, j_elem) 四元组对 edit edge 加 motif-aware weight.

    rules: list of (target_cls, pred_cls, i_z, j_z, weight)
        - i_z/j_z 为 None 时表示对该 i/j 元素无约束（兜底规则）
        - weight 为乘数，对每条命中规则的 edge 累乘 (即多规则命中时取累乘 weight)

    Returns: per-edge weight Tensor [K_total], 默认 1.0
    """
    K = edit_logits.size(0)
    if K == 0 or atom_z is None or atom_z.numel() == 0:
        return torch.ones(K, device=edit_logits.device, dtype=torch.float32)

    device = edit_logits.device
    weights = torch.ones(K, device=device, dtype=torch.float32)
    pred_cls = edit_logits.argmax(dim=-1)
    i_idx = all_cand_global[:, 0].long()
    j_idx = all_cand_global[:, 1].long()
    i_elem = atom_z[i_idx]
    j_elem = atom_z[j_idx]

    for rule in rules:

        tgt, prd, iz, jz, w = rule
        mask = (edit_labels == tgt) & (pred_cls == prd)
        if iz is not None and jz is not None:

            pair_match = ((i_elem == iz) & (j_elem == jz)) | (
                (i_elem == jz) & (j_elem == iz)
            )
            mask = mask & pair_match

        if mask.any():
            weights[mask] = weights[mask] * float(w)
    return weights

def compute_cc_pair_constraint(rc_probs, all_cand_global, edit_labels, atom_z):
    """v20.2: C-C pair-level constraint. 对 GT edit 中的 C-C 键对要求 RC pred 一致.

    针对 ep36 supplementary S1 发现 C-C edit acc=76.04% missing=22.16% 远差于杂原子键.
    通过约束 |rc_prob[i] - rc_prob[j]| 在 GT C-C edge 上接近 0, 强化 C-C pair 的协同识别.

    Returns: scalar loss (mean over qualifying edges); 若无 qualifying edge 返回 0.
    """
    if all_cand_global.size(0) == 0 or atom_z is None or atom_z.numel() == 0:
        return torch.tensor(0.0, device=rc_probs.device, dtype=torch.float32)

    i_idx = all_cand_global[:, 0].long()
    j_idx = all_cand_global[:, 1].long()
    i_elem = atom_z[i_idx]
    j_elem = atom_z[j_idx]

    mask = (i_elem == 6) & (j_elem == 6) & (edit_labels != 0)
    if not mask.any():
        return torch.tensor(0.0, device=rc_probs.device, dtype=torch.float32)
    p_i = rc_probs[i_idx[mask]]
    p_j = rc_probs[j_idx[mask]]
    return (p_i - p_j).abs().mean()

def focal_bce_with_logits(logits, target_smooth, target_binary,
                          pos_weight, gamma, gamma_neg=None, reduction='none'):
    """v9: Focal BCE for RC.  v14: 支持非对称 gamma.

    target_smooth: 平滑后的 0/1 软标签 (用于 BCE 计算)
    target_binary: 0/1 硬标签 (用于决定 pt = p 还是 1-p)
    gamma: 正样本 focal 聚焦参数；gamma=0 退化为标准 BCE
    gamma_neg: 负样本 focal 聚焦参数（None 时退化为对称 gamma）

    v14 非对称设计:
      正样本 gamma=2.5 — 更强聚焦 hard FN（prob<0.2 的真 RC 原子）
      负样本 gamma_neg=1.5 — 保留 hard FP 梯度，防止 focal 过度压制假阳性的学习信号
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, target_smooth, pos_weight=pos_weight, reduction='none'
    )
    if gamma > 0:
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pt = torch.where(target_binary > 0.5, p, 1 - p)
            if gamma_neg is not None:
                g = torch.where(target_binary > 0.5, gamma, gamma_neg)
            else:
                g = gamma
            focal = (1 - pt).clamp(min=1e-6) ** g
        bce = focal * bce
    if reduction == 'mean':
        return bce.mean()
    if reduction == 'sum':
        return bce.sum()
    return bce

def compute_conservation_loss(outputs, batch, config):
    """v19 V2 (§32.4) Phase 1 部分: S1 价态软上限 + S3 pair-exist 一致性.

    Returns:
        (loss_valence_rc, loss_pair_exist) 都是标量 tensor
    """
    rc_logits = outputs["rc_logits"]
    rc_probs = outputs.get("rc_probs")
    if rc_probs is None:
        rc_probs = torch.sigmoid(rc_logits)
    device = rc_logits.device

    loss_valence_rc = torch.tensor(0.0, device=device)
    if config.get("use_valence_rc_penalty", False):
        atom_z = batch.x[:, 0].long().clamp(0, ATOM_MAX_VALENCE.size(0) - 1)
        max_val = ATOM_MAX_VALENCE.to(device)[atom_z].to(rc_probs.dtype)
        node_deg = compute_node_bond_sum(batch.edge_index, batch.edge_attr, rc_probs.size(0)).to(rc_probs.dtype)
        remaining = (max_val - node_deg).clamp(min=0.0)
        cap_sat = float(config.get("valence_rc_cap_saturated", 0.3))
        v_cap = torch.where(remaining < 0.5, torch.full_like(remaining, cap_sat),
                             torch.ones_like(remaining))
        over = F.relu(rc_probs - v_cap)
        loss_valence_rc = (over ** 2).mean()

    loss_pair_exist = torch.tensor(0.0, device=device)
    if config.get("use_pair_exist_constraint", False):
        hard_rc_mask = rc_probs > 0.5
        if hard_rc_mask.any():
            partner_max = compute_partner_max_prob(rc_probs, batch.batch)
            margin = float(config.get("pair_exist_margin", 0.2))
            gap = rc_probs[hard_rc_mask] - partner_max[hard_rc_mask]
            hinge = F.relu(gap - margin)
            if hinge.numel() > 0:
                loss_pair_exist = (hinge ** 2).mean()

    return loss_valence_rc, loss_pair_exist

def compute_valence_dh_loss(outputs, batch, config):
    """v24 §47: Valence Consistency Loss (基于 y_dh 真实化学价守恒)

    数学定义 (对每个 atom_filter 选中的原子 i):
        pred_dvalence(i) = Σ_{j ∈ cand_edges_of_i} E[Δbond | softmax(edit_logits[i,j])] + y_dh(i)
        L_valence_dh = mean_i [ pred_dvalence(i) ** 2 ]

    其中 E[Δbond | softmax] = Σ_c softmax(logits)[c] * CLASS_TO_DELTA[c]
    取 use_softmax=False 则用 argmax 直通 (无梯度通向 edit_logits, 仅作消融)

    关键约束:
      - cand_edges 是单向边列表 [E, 2] (i<j 或 i>j 都可能), 必须双向计入到 atom 的 dvalence
      - 仅对 GT RC 原子 / pred_rc>0.5 / 全部原子计算 (atom_filter)
      - LG 原子 y_dh=0, 其 cand 边自然对应 0 期望, 不污染 loss

    返回:
      loss_valence_dh: 标量 tensor (无梯度路径时为 detach 的 0)
    """
    device = outputs["rc_logits"].device
    edit_logits = outputs.get("edit_logits", None)
    cand_edges_list = outputs.get("cand_edges", None)
    if edit_logits is None or cand_edges_list is None or edit_logits.size(0) == 0:
        return torch.tensor(0.0, device=device)

    y_dh_list = getattr(batch, "y_dh_list", None)
    if y_dh_list is None or len(y_dh_list) == 0:
        return torch.tensor(0.0, device=device)

    rc_target_flat = outputs.get("rc_target_flat", None)
    rc_probs = outputs.get("rc_probs", None)
    batch_idx = batch.batch
    n_mols = len(cand_edges_list)
    node_counts = torch.bincount(batch_idx, minlength=n_mols)
    node_offsets = torch.cat([torch.tensor([0], device=device), node_counts.cumsum(0)[:-1]])

    atom_filter = config.get("valence_dh_atom_filter", "gt_rc")
    use_softmax = config.get("valence_dh_use_softmax", True)

    delta_lookup = CLASS_TO_DELTA.to(device=device, dtype=edit_logits.dtype)

    if use_softmax:
        edit_probs = torch.softmax(edit_logits.float(), dim=-1)
        e_delta = (edit_probs * delta_lookup.float().unsqueeze(0)).sum(dim=-1)
    else:
        pred_cls = edit_logits.argmax(dim=-1)
        e_delta = delta_lookup.float()[pred_cls]

    sq_sum = torch.tensor(0.0, device=device)
    cnt = 0
    edge_offset = 0
    for mol_idx in range(n_mols):
        cand = cand_edges_list[mol_idx]
        k_i = cand.size(0)
        n_atoms = int(node_counts[mol_idx].item())

        if atom_filter == "all":
            sel_mask = torch.ones(n_atoms, dtype=torch.bool, device=device)
        elif atom_filter == "pred_rc_thr05":
            start = int(node_offsets[mol_idx].item())
            sel_mask = (rc_probs[start:start + n_atoms] > 0.5)
        else:
            if rc_target_flat is None:
                edge_offset += k_i
                continue
            start = int(node_offsets[mol_idx].item())
            sel_mask = (rc_target_flat[start:start + n_atoms] > 0.5)

        if not sel_mask.any():
            edge_offset += k_i
            continue

        atom_dbond = torch.zeros(n_atoms, device=device, dtype=e_delta.dtype)
        if k_i > 0:
            mol_e_delta = e_delta[edge_offset:edge_offset + k_i]
            src = cand[:, 0].long().to(device)
            dst = cand[:, 1].long().to(device)
            atom_dbond.scatter_add_(0, src, mol_e_delta)
            atom_dbond.scatter_add_(0, dst, mol_e_delta)

        y_dh = y_dh_list[mol_idx].to(device=device, dtype=atom_dbond.dtype)
        if y_dh.size(0) != n_atoms:
            edge_offset += k_i
            continue
        dvalence = atom_dbond + y_dh

        sq_sum = sq_sum + (dvalence[sel_mask] ** 2).sum()
        cnt += int(sel_mask.sum().item())

        edge_offset += k_i

    if cnt == 0:
        return torch.tensor(0.0, device=device)
    return sq_sum / cnt

def compute_fukui_rank_loss(fukui, rc_target, global_frag_id, num_global_frags, margin=0.01):
    """v19 V6 (§32.8): frag-内 RC 原子的 fukui_abs 应高于非 RC 原子.

    Hinge: mean(ReLU(mean_nonrc - mean_rc + margin)) over mixed frags.
    """
    if fukui is None or fukui.numel() == 0 or num_global_frags == 0:
        return torch.tensor(0.0, device=rc_target.device if rc_target is not None else "cpu")
    device = fukui.device
    fukui_f = fukui.to(torch.float32)
    fukui_abs = fukui_f.abs().max(dim=-1)[0]
    is_rc = rc_target.to(torch.bool)
    gfid = global_frag_id.to(torch.long)
    ng = int(num_global_frags)

    ones = torch.ones(gfid.size(0), device=device, dtype=torch.float32)
    n_per_frag = torch.zeros(ng, device=device, dtype=torch.float32)
    n_per_frag.scatter_add_(0, gfid, ones)
    n_rc_per_frag = torch.zeros(ng, device=device, dtype=torch.float32)
    n_rc_per_frag.scatter_add_(0, gfid, is_rc.to(torch.float32))
    mixed_mask = (n_rc_per_frag > 0) & (n_rc_per_frag < n_per_frag)

    sum_rc = torch.zeros(ng, device=device, dtype=torch.float32)
    sum_rc.scatter_add_(0, gfid, fukui_abs * is_rc.to(torch.float32))
    sum_nonrc = torch.zeros(ng, device=device, dtype=torch.float32)
    sum_nonrc.scatter_add_(0, gfid, fukui_abs * (~is_rc).to(torch.float32))
    n_nonrc_per_frag = n_per_frag - n_rc_per_frag

    mean_rc = sum_rc / (n_rc_per_frag + 1e-6)
    mean_nonrc = sum_nonrc / (n_nonrc_per_frag + 1e-6)

    hinge = F.relu(mean_nonrc - mean_rc + margin)
    if mixed_mask.any():
        return hinge[mixed_mask].mean()
    return torch.tensor(0.0, device=device)

def compute_loss(
    outputs,
    batch,
    config,
    current_phase,
    edit_ramp=1.0,
    model=None,
    epoch=None,
    contrastive_loss_fn=None,
):
    """v18 Phase 1 loss:
      L_rc        = focal BCE (+ element reweight + hot-bin + curriculum hard-neg + top-K FP)
      L_all_hit   = per-reaction all-atom-hit (v18 N2)
      L_pair_hard = pair-level hard BCE (v18 N1, hard supervision from y_delta)
      L_pair_sym  = v16 V2 对称性软约束（降权 0.15）
      L_pair_cont = v13 InfoNCE（降权 0.05）
      L_bond_rc   = bond-level（降权 0.15）
    """
    device = outputs["rc_logits"].device
    batch_idx = batch.batch

    rc_target_flat = outputs["rc_target_flat"]

    rc_ls = config.get("rc_label_smoothing", 0.0)
    if rc_ls > 0:
        rc_target_smooth = (
            rc_target_flat * (1.0 - rc_ls) + rc_ls * 0.5
        ).clamp(0.0, 1.0)
    else:
        rc_target_smooth = rc_target_flat

    pw = (
        config["phase2_rc_pos_weight"]
        if current_phase >= 2
        else config["rc_pos_weight"]
    )
    pos_weight = torch.tensor([pw], device=device)
    rc_logits_f = outputs["rc_logits"].float()

    rc_focal_gamma = (
        config.get("rc_focal_phase2_gamma", 0.5)
        if current_phase >= 2
        else config.get("rc_focal_gamma", 0.0)
    )

    rc_focal_gamma_neg = config.get("rc_focal_gamma_neg", None)
    loss_rc_per = focal_bce_with_logits(
        rc_logits_f, rc_target_smooth, rc_target_flat,
        pos_weight=pos_weight, gamma=rc_focal_gamma,
        gamma_neg=rc_focal_gamma_neg, reduction='none'
    )

    if config.get("rc_element_reweight", False):
        atom_elem = batch.x[:, 0].long()
        atom_hyb = batch.x[:, 3].long()
        atom_arom = batch.x[:, 4].long()
        elem_weight = torch.ones_like(rc_target_flat)
        pos_mask_t = rc_target_flat > 0.5
        neg_mask_t = ~pos_mask_t

        f_pos = (atom_elem == 9) & pos_mask_t
        elem_weight[f_pos] = config.get("rc_elem_w_F_pos", 2.0)

        heavy_halide = (atom_elem == 17) | (atom_elem == 35) | (atom_elem == 53)
        elem_weight[heavy_halide & neg_mask_t] = config.get(
            "rc_elem_w_halide_neg", 3.0
        )

        p_neg = (atom_elem == 15) & neg_mask_t
        elem_weight[p_neg] = config.get("rc_elem_w_P_neg", 5.0)

        br_neg = (atom_elem == 35) & neg_mask_t
        elem_weight[br_neg] = config.get("rc_elem_w_Br_neg", 2.5)

        is_C = (atom_elem == 6)
        c_arom_pos = is_C & (atom_arom == 1) & pos_mask_t
        c_alip_pos = is_C & (atom_arom == 0) & pos_mask_t
        elem_weight[c_arom_pos] = config.get("rc_elem_w_C_arom_pos", 1.3)
        elem_weight[c_alip_pos] = config.get("rc_elem_w_C_alip_pos", 1.3)

        loss_rc_per = loss_rc_per * elem_weight

    if config.get("rc_hot_bin_reweight", False):

        bs = int(batch_idx.max().item()) + 1
        mol_sizes = torch.bincount(batch_idx, minlength=bs).float()
        rc_per_mol = torch.zeros(bs, device=device)
        rc_per_mol.scatter_add_(0, batch_idx, rc_target_flat)
        density = rc_per_mol / mol_sizes.clamp(min=1.0)

        w_size = torch.ones(bs, device=device)
        w_size = torch.where(mol_sizes > 100.0, torch.full_like(w_size, 2.5), w_size)
        w_size = torch.where((mol_sizes > 60.0) & (mol_sizes <= 100.0), torch.full_like(w_size, 2.0), w_size)
        w_size = torch.where((mol_sizes > 40.0) & (mol_sizes <= 60.0), torch.full_like(w_size, 1.5), w_size)

        w_density = 1.0 + (density - 0.10).clamp(min=0.0) * 5.0
        mol_weight = w_size * w_density
        atom_mol_weight = mol_weight[batch_idx]
        loss_rc_per = loss_rc_per * atom_mol_weight

    if (config.get("use_rc_main_weight", False)
        and current_phase >= 2
        and (epoch is None or epoch >= config.get("rc_main_warmup_epochs", 3))
        and getattr(batch, "main_atom_mask", None) is not None):
        try:
            main_mask = batch.main_atom_mask.to(device)
            if main_mask.numel() == loss_rc_per.numel():
                is_pos = rc_target_flat > 0.5
                rc_main_mult_val = float(config.get("rc_main_loss_mult", 1.5))
                rc_non_main_mult_val = float(config.get("rc_non_main_loss_mult", 0.3))
                rc_w = torch.where(
                    main_mask | is_pos,
                    torch.full_like(rc_target_flat, rc_main_mult_val),
                    torch.full_like(rc_target_flat, rc_non_main_mult_val),
                )
                loss_rc_per = loss_rc_per * rc_w
        except Exception:
            pass

    sample_weight = torch.ones_like(rc_target_flat)
    rc_probs_detach = None

    hn_ratio_start = config.get("rc_hard_neg_ratio_start", config.get("rc_hard_neg_ratio", 0.3))
    hn_ratio_end = config.get("rc_hard_neg_ratio_end", hn_ratio_start)
    hn_weight_start = config.get("rc_hard_neg_weight", 5.0)
    hn_weight_end = config.get("rc_hard_neg_weight_end", hn_weight_start)
    hn_curriculum_epochs = config.get("rc_hard_neg_curriculum_epochs", 30)
    hn_warmup = config.get("rc_neg_weight_warmup_epochs", 10)
    if hn_ratio_start > 0 and hn_weight_start > 1.0:
        neg_mask = rc_target_flat == 0
        neg_count = neg_mask.sum().item()
        if neg_count > 0:
            ep = epoch or 0

            warmup_progress = min(1.0, ep / max(hn_warmup, 1))

            curriculum_progress = min(1.0, ep / max(hn_curriculum_epochs, 1))
            hn_ratio = hn_ratio_start - (hn_ratio_start - hn_ratio_end) * curriculum_progress
            hn_weight = 1.0 + (hn_weight_start - 1.0) * warmup_progress
            hn_weight = hn_weight + (hn_weight_end - hn_weight_start) * curriculum_progress

            rc_probs_detach = rc_logits_f.detach().sigmoid()
            neg_probs = rc_probs_detach[neg_mask]

            k = max(1, int(neg_count * hn_ratio))
            threshold = neg_probs.topk(k).values[-1]
            hard_neg_mask = neg_mask & (rc_probs_detach >= threshold)
            sample_weight[hard_neg_mask] = hn_weight

    if (
        config.get("use_topk_fp_hard_neg", False)
        and (epoch is None or epoch >= config.get("topk_fp_warmup_epochs", 2))
    ):
        with torch.no_grad():
            if rc_probs_detach is None:
                rc_probs_detach = rc_logits_f.detach().sigmoid()

            masked_probs = rc_probs_detach.clone()
            masked_probs[rc_target_flat.bool()] = -1.0
            bs_fp = int(batch_idx.max().item()) + 1
            K_fp = int(config.get("topk_fp_per_reaction", 5))
            topk_fp_mask = torch.zeros_like(rc_target_flat, dtype=torch.bool)
            for b in range(bs_fp):
                mask_b = (batch_idx == b) & (rc_target_flat == 0)
                n_b = int(mask_b.sum().item())
                if n_b < K_fp:
                    continue
                probs_b = masked_probs[mask_b]
                thr_b = probs_b.topk(K_fp).values[-1]
                topk_fp_mask |= (mask_b & (masked_probs >= thr_b))
        w_topk_fp = config.get("w_topk_fp_hard_neg", 2.5)

        sample_weight[topk_fp_mask] = torch.clamp(
            sample_weight[topk_fp_mask], min=w_topk_fp
        )

    loss_rc = (loss_rc_per * sample_weight).mean()

    loss_pair_sym = torch.tensor(0.0, device=device)
    pair_sym_warmup = config.get("pair_sym_warmup_epochs", 5)
    if (
        config.get("w_pair_sym", 0.0) > 0
        and outputs.get("rc_pair_index") is not None
        and outputs["rc_pair_index"].numel() > 0
        and (epoch is None or epoch >= pair_sym_warmup)
    ):
        pair_idx = outputs["rc_pair_index"]
        p_all = torch.sigmoid(rc_logits_f)
        p_i = p_all[pair_idx[:, 0]]
        p_j = p_all[pair_idx[:, 1]]

        margin = config.get("pair_sym_margin", 0.7)
        margin_loss_i = F.relu(margin - p_i)
        margin_loss_j = F.relu(margin - p_j)

        sym_term = (p_i - p_j).abs() * config.get("pair_sym_alpha", 1.0)
        loss_pair_sym = (margin_loss_i + margin_loss_j + sym_term).mean()

    loss_pair_contrastive = torch.tensor(0.0, device=device)
    pc_warmup = config.get("pair_contrastive_warmup_epochs", 12)
    if (
        config.get("w_pair_contrastive", 0.0) > 0
        and outputs.get("rc_pair_index") is not None
        and outputs["rc_pair_index"].numel() > 0
        and outputs.get("h_rc") is not None
        and (epoch is None or epoch >= pc_warmup)
    ):
        pair_idx = outputs["rc_pair_index"]
        h_rc_all = outputs["h_rc"].float()
        tau = config.get("pair_contrastive_temp", 0.1)

        z = F.normalize(h_rc_all, dim=-1)
        z_i = z[pair_idx[:, 0]]
        z_j = z[pair_idx[:, 1]]

        pos_sim = (z_i * z_j).sum(dim=-1) / tau

        sim_matrix_i = z_i @ z.t() / tau
        sim_matrix_j = z_j @ z.t() / tau

        log_denom_i = torch.logsumexp(sim_matrix_i, dim=-1)
        log_denom_j = torch.logsumexp(sim_matrix_j, dim=-1)
        loss_pair_contrastive = (
            (log_denom_i - pos_sim).mean() + (log_denom_j - pos_sim).mean()
        ) * 0.5

    loss_bond_rc = torch.tensor(0.0, device=device)
    if (
        config.get("w_bond_rc", 0.0) > 0
        and outputs.get("bond_logits") is not None
        and outputs["bond_logits"].numel() > 0
    ):

        ei = batch.edge_index
        keep_mask = ei[0] < ei[1]
        if keep_mask.any():
            bond_logits_f = outputs["bond_logits"].float()[keep_mask]
            src_undir = ei[0][keep_mask]
            dst_undir = ei[1][keep_mask]

            num_atoms_total = rc_target_flat.size(0)
            bond_label = torch.zeros_like(bond_logits_f)
            if outputs.get("rc_pair_index") is not None and outputs["rc_pair_index"].numel() > 0:
                pair_idx = outputs["rc_pair_index"]

                rc_keys = pair_idx[:, 0] * num_atoms_total + pair_idx[:, 1]
                bond_keys = src_undir * num_atoms_total + dst_undir

                mask_in = torch.isin(bond_keys, rc_keys)
                bond_label[mask_in] = 1.0
            bond_pw = torch.tensor(
                [config.get("bond_rc_pos_weight", 10.0)], device=device
            )
            loss_bond_rc = F.binary_cross_entropy_with_logits(
                bond_logits_f, bond_label, pos_weight=bond_pw
            )

    loss_pair_head = torch.tensor(0.0, device=device)
    loss_pair_teach = torch.tensor(0.0, device=device)
    if (
        config.get("w_pair_head", 0.0) > 0
        and outputs.get("pair_logits") is not None
        and outputs["pair_logits"].numel() > 0
        and outputs.get("pair_cand_index") is not None
        and outputs.get("pair_targets") is not None
    ):
        p_logits = outputs["pair_logits"].float()
        p_targets = outputs["pair_targets"].float()
        p_idx = outputs["pair_cand_index"]

        pair_head_pw = torch.tensor(
            [config.get("pair_head_pos_weight", 5.0)], device=device
        )
        loss_pair_head = F.binary_cross_entropy_with_logits(
            p_logits, p_targets, pos_weight=pair_head_pw
        )

        pair_head_warmup = config.get("pair_head_warmup_epochs", 5)
        if epoch is None or epoch >= pair_head_warmup:
            pair_prob = torch.sigmoid(p_logits).detach()
            atom_i_prob = torch.sigmoid(rc_logits_f[p_idx[:, 0]])
            atom_j_prob = torch.sigmoid(rc_logits_f[p_idx[:, 1]])

            pos_mask_pair = p_targets.bool()
            neg_mask_pair = ~pos_mask_pair

            if pos_mask_pair.any():
                if config.get("pair_teacher_asymmetric", False):
                    margin_teach = config.get("pair_teacher_margin", 0.05)
                    p_pair_pos = pair_prob[pos_mask_pair]
                    gap_i = (p_pair_pos - atom_i_prob[pos_mask_pair] - margin_teach).clamp(min=0.0)
                    gap_j = (p_pair_pos - atom_j_prob[pos_mask_pair] - margin_teach).clamp(min=0.0)
                    l_teach_pos = 0.5 * (gap_i.pow(2).mean() + gap_j.pow(2).mean())
                else:
                    l_teach_pos = (
                        F.mse_loss(atom_i_prob[pos_mask_pair], pair_prob[pos_mask_pair])
                      + F.mse_loss(atom_j_prob[pos_mask_pair], pair_prob[pos_mask_pair])
                    ) * 0.5
            else:
                l_teach_pos = p_logits.new_zeros(())

            if neg_mask_pair.any():
                sym_penalty = (
                    atom_i_prob[neg_mask_pair] - atom_j_prob[neg_mask_pair]
                ).abs().mean()
            else:
                sym_penalty = p_logits.new_zeros(())

            teach_neg_w = config.get("pair_teacher_neg_weight", 0.3)
            loss_pair_teach = l_teach_pos + teach_neg_w * sym_penalty

    if current_phase >= 2 and outputs["edit_logits"].size(0) > 0:
        target_alpha = torch.tensor(config["focal_alpha"], device=device)
        alpha_progress = min(
            1.0,
            edit_ramp
            * config["phase2_warmup_epochs"]
            / max(config["focal_alpha_warmup_epochs"], 1),
        )
        current_alpha = 1.0 + (target_alpha - 1.0) * alpha_progress

        motif_warmup = config.get("motif_hard_neg_warmup_epochs", 15)
        use_motif = (
            config.get("use_motif_hard_neg", False)
            and (epoch is None or epoch >= motif_warmup)
            and outputs.get("all_cand_global") is not None
            and outputs["all_cand_global"].size(0) > 0
        )

        use_K_pos = (
            config.get("use_K_pos_edge_weight", False)
            and (epoch is None or epoch >= config.get("K_pos_edge_warmup_epochs", 2))
            and outputs.get("all_cand_global") is not None
            and outputs["all_cand_global"].size(0) > 0
            and getattr(batch, "precomputed_rc_target_flat", None) is not None
            and getattr(batch, "batch", None) is not None
        )

        if use_motif or use_K_pos:
            focal_per_sample = focal_loss(
                outputs["edit_logits"],
                outputs["edit_labels"],
                gamma=config["focal_gamma"],
                alpha=current_alpha,
                label_smoothing=0.05,
                reduction="none",
            )
            edge_w = torch.ones_like(focal_per_sample)

            if use_motif:
                atom_z = batch.x[:, 0].long()
                motif_w = apply_motif_hard_neg_weights(
                    outputs["edit_logits"],
                    outputs["edit_labels"],
                    outputs["all_cand_global"],
                    atom_z,
                    config["motif_hard_neg_rules"],
                )
                edge_w = edge_w * motif_w

            if use_K_pos:
                rc_target = batch.precomputed_rc_target_flat.float()
                mol_idx_atom = batch.batch
                n_mols = int(mol_idx_atom.max().item()) + 1
                K_per_mol = torch.zeros(n_mols, device=device, dtype=torch.float)
                K_per_mol.scatter_add_(0, mol_idx_atom, rc_target)
                K_per_mol = K_per_mol.long()
                cand_global = outputs["all_cand_global"]
                edge_mol = mol_idx_atom[cand_global[:, 0]]
                edge_K = K_per_mol[edge_mol]
                is_positive = (outputs["edit_labels"] != 0)
                K4_mult = float(config.get("K_pos_edge_K4_mult", 1.5))
                K5p_mult = float(config.get("K_pos_edge_K5p_mult", 2.0))
                K_pos_w = torch.ones_like(edge_K, dtype=torch.float)
                K_pos_w = torch.where(
                    is_positive & (edge_K == 4),
                    torch.full_like(K_pos_w, K4_mult),
                    K_pos_w,
                )
                K_pos_w = torch.where(
                    is_positive & (edge_K >= 5),
                    torch.full_like(K_pos_w, K5p_mult),
                    K_pos_w,
                )
                edge_w = edge_w * K_pos_w

            loss_edit = (focal_per_sample * edge_w).mean()
        else:
            loss_edit = focal_loss(
                outputs["edit_logits"],
                outputs["edit_labels"],
                gamma=config["focal_gamma"],
                alpha=current_alpha,
                label_smoothing=0.05,
            )
        loss_edit = torch.clamp(loss_edit, max=config["edit_loss_clamp"])
    else:
        loss_edit = torch.tensor(0.0, device=device)

    loss_cc_pair = torch.tensor(0.0, device=device)
    cc_pair_warmup = config.get("cc_pair_warmup_epochs", 20)
    if (
        config.get("use_cc_pair_constraint", False)
        and current_phase >= 2
        and outputs["edit_logits"].size(0) > 0
        and outputs.get("all_cand_global") is not None
        and outputs["all_cand_global"].size(0) > 0
        and (epoch is None or epoch >= cc_pair_warmup)
    ):
        atom_z_cc = batch.x[:, 0].long()
        loss_cc_pair = compute_cc_pair_constraint(
            outputs["rc_probs"],
            outputs["all_cand_global"],
            outputs["edit_labels"],
            atom_z_cc,
        )

    loss_all_atom_hit = torch.tensor(0.0, device=device)
    all_hit_warmup = config.get("all_atom_hit_warmup_epochs", 2)
    if (
        config.get("use_all_atom_hit_loss", False)
        and config.get("w_all_atom_hit", 0.0) > 0
        and (epoch is None or epoch >= all_hit_warmup)
    ):
        loss_all_atom_hit = compute_all_atom_hit_loss(
            rc_logits_f, rc_target_flat, batch_idx
        )

    loss_pair_hard = torch.tensor(0.0, device=device)
    pair_hard_warmup = config.get("pair_hard_warmup_epochs", 3)
    if (
        config.get("use_pair_hard_head", False)
        and config.get("w_pair_hard", 0.0) > 0
        and outputs.get("pair_hard_logits") is not None
        and outputs["pair_hard_logits"].numel() > 0
        and (epoch is None or epoch >= pair_hard_warmup)
    ):
        ph_logits = outputs["pair_hard_logits"].float()
        ph_target = outputs["pair_hard_target"].float()
        ph_pw = torch.tensor(
            [config.get("pair_hard_pos_weight", 3.0)], device=device
        )
        loss_pair_hard = F.binary_cross_entropy_with_logits(
            ph_logits, ph_target, pos_weight=ph_pw
        )

    loss_valence_rc = torch.tensor(0.0, device=device)
    loss_pair_exist = torch.tensor(0.0, device=device)
    pair_exist_warmup = config.get("pair_exist_warmup_epochs", 0)
    valence_warmup = config.get("valence_rc_warmup_epochs", 0)
    if (config.get("use_valence_rc_penalty", False) or config.get("use_pair_exist_constraint", False))            and (epoch is None or epoch >= max(pair_exist_warmup, valence_warmup) or True):

        loss_valence_rc_raw, loss_pair_exist_raw = compute_conservation_loss(outputs, batch, config)
        if epoch is None or epoch >= valence_warmup:
            loss_valence_rc = loss_valence_rc_raw
        if epoch is None or epoch >= pair_exist_warmup:
            loss_pair_exist = loss_pair_exist_raw

    loss_fukui_rank = torch.tensor(0.0, device=device)
    fukui_rank_warmup = config.get("fukui_rank_warmup_epochs", 0)
    if (
        config.get("use_fukui_rank_loss", False)
        and outputs.get("fukui") is not None
        and outputs.get("global_frag_id") is not None
        and rc_target_flat is not None
        and (epoch is None or epoch >= fukui_rank_warmup)
    ):
        loss_fukui_rank = compute_fukui_rank_loss(
            outputs["fukui"], rc_target_flat,
            outputs["global_frag_id"], outputs["num_global_frags"],
            margin=float(config.get("fukui_rank_margin", 0.01)),
        )

    loss_mol_aux = torch.tensor(0.0, device=device)
    if (
        config.get("use_mol_aux_loss", False)
        and rc_target_flat is not None
        and outputs.get("batch_idx") is not None
    ):
        loss_mol_aux = compute_mol_aux_loss(
            outputs["rc_probs"].float(),
            rc_target_flat.float(),
            outputs["batch_idx"],
            temperature_atom=float(config.get("mol_aux_temperature_atom", 0.02)),
            clean_threshold=float(config.get("mol_aux_clean_threshold", 0.95)),
        )

    loss_set_recall = torch.tensor(0.0, device=device)
    loss_set_precision = torch.tensor(0.0, device=device)
    set_warmup = config.get("set_loss_warmup_epochs", 3)
    set_active = (
        current_phase >= 2
        and (epoch is None or epoch >= set_warmup)
        and rc_target_flat is not None
    )

    set_k_weights = config.get("set_loss_k_weights", {})
    if set_active and config.get("use_set_recall_loss", False):
        loss_set_recall = compute_set_recall_loss(
            outputs["rc_logits"],
            rc_target_flat,
            batch_idx,
            set_k_weights,
            p_clamp_min=float(config.get("set_loss_p_clamp_min", 1e-6)),
            max_clamp=float(config.get("set_loss_max_clamp", 5.0)),
        )
    if set_active and config.get("use_set_precision_loss", False):
        loss_set_precision = compute_set_precision_loss(
            outputs["rc_logits"],
            rc_target_flat,
            batch_idx,
            set_k_weights,
            min_K=int(config.get("set_precision_min_K", 3)),
            p_clamp_min=float(config.get("set_loss_p_clamp_min", 1e-6)),
            max_clamp=float(config.get("set_loss_max_clamp", 5.0)),
        )

    loss_pointer_rc = torch.tensor(0.0, device=device)
    if (
        config.get("use_pointer_rc_loss", False)
        and current_phase >= 2
        and rc_target_flat is not None
    ):
        loss_pointer_rc = compute_mol_pointer_rc_loss(
            outputs["rc_logits"],
            rc_target_flat,
            batch_idx,
        )

    loss_valence_dh = torch.tensor(0.0, device=device)
    if (
        config.get("use_valence_dh", False)
        and current_phase >= 2
        and outputs.get("edit_logits", None) is not None
        and outputs["edit_logits"].size(0) > 0
    ):
        warmup_ep = int(config.get("valence_dh_warmup_epochs", 0))
        if epoch is None or int(epoch) > warmup_ep:
            loss_valence_dh = compute_valence_dh_loss(outputs, batch, config)

    if (
        config.get("use_main_edit_weight", False)
        and current_phase >= 2
        and outputs["edit_logits"].size(0) > 0
        and outputs.get("all_cand_global") is not None
        and outputs["all_cand_global"].size(0) > 0
        and (epoch is None or epoch >= config.get("main_edit_warmup_epochs", 3))
        and getattr(batch, "main_atom_mask", None) is not None
    ):
        main_mask_global = batch.main_atom_mask
        cand_global = outputs["all_cand_global"]
        cand_i_main = main_mask_global[cand_global[:, 0]]
        cand_j_main = main_mask_global[cand_global[:, 1]]
        any_main = cand_i_main | cand_j_main
        main_mult = float(config.get("main_edit_loss_mult", 1.5))
        non_main_mult = float(config.get("non_main_edit_loss_mult", 0.5))
        main_w = torch.where(
            any_main,
            torch.full_like(any_main, main_mult, dtype=torch.float),
            torch.full_like(any_main, non_main_mult, dtype=torch.float),
        )

        if outputs.get("edit_labels") is not None and outputs["edit_logits"].size(0) == main_w.size(0):
            target_alpha_mw = torch.tensor(config["focal_alpha"], device=device)
            alpha_progress_mw = min(
                1.0,
                edit_ramp * config["phase2_warmup_epochs"]
                / max(config["focal_alpha_warmup_epochs"], 1),
            )
            current_alpha_mw = 1.0 + (target_alpha_mw - 1.0) * alpha_progress_mw
            focal_pers = focal_loss(
                outputs["edit_logits"],
                outputs["edit_labels"],
                gamma=config["focal_gamma"],
                alpha=current_alpha_mw,
                label_smoothing=0.05,
                reduction="none",
            )
            loss_edit = (focal_pers * main_w).mean()
            loss_edit = torch.clamp(loss_edit, max=config["edit_loss_clamp"])

    loss_margin_penalty = torch.tensor(0.0, device=device)
    loss_mutual_exclusion = torch.tensor(0.0, device=device)
    loss_edit_conditional_rc = torch.tensor(0.0, device=device)
    loss_l1_sparsity = torch.tensor(0.0, device=device)

    _v25_phase2_active = (
        current_phase >= 2
        and outputs["edit_logits"].size(0) > 0
        and outputs.get("edit_labels") is not None
    )
    if _v25_phase2_active:
        edit_logits_v25 = outputs["edit_logits"].float()
        edit_labels_v25 = outputs["edit_labels"]

        if config.get("use_margin_penalty", False):
            wu_end = int(config.get("margin_penalty_warmup_end", 5))
            if epoch is None or int(epoch) >= wu_end:
                nz_mask_v25 = edit_labels_v25 != 0
                if nz_mask_v25.any():
                    probs_v25 = torch.softmax(edit_logits_v25[nz_mask_v25], dim=-1)
                    p_gt = probs_v25.gather(
                        -1, edit_labels_v25[nz_mask_v25].unsqueeze(-1)
                    ).squeeze(-1)
                    p_0 = probs_v25[:, 0]
                    margin_v25 = p_gt - p_0
                    loss_margin_penalty = -margin_v25.clamp(min=0).mean()

        if (
            config.get("use_mutual_exclusion", False)
            and outputs.get("all_cand_global") is not None
            and outputs["all_cand_global"].size(0) > 0
        ):
            wu_end = int(config.get("mutual_exclusion_warmup_end", 5))
            if epoch is None or int(epoch) >= wu_end:
                from torch_scatter import scatter_logsumexp, scatter_max
                cand_global_v25 = outputs["all_cand_global"]
                n_total_atoms = batch.x.size(0)
                min_edges = int(config.get("mutual_exclusion_min_edges", 2))
                plus_one_logit = edit_logits_v25[:, 1]

                vec_losses = []
                for a_col in (0, 1):
                    a_idx = cand_global_v25[:, a_col]
                    counts = torch.bincount(a_idx, minlength=n_total_atoms)
                    elig_mask = counts >= min_edges
                    if not elig_mask.any():
                        continue
                    lse = scatter_logsumexp(
                        plus_one_logit, a_idx, dim=0, dim_size=n_total_atoms)
                    mx, _ = scatter_max(
                        plus_one_logit, a_idx, dim=0, dim_size=n_total_atoms)

                    softmax_max = (mx - lse).exp()
                    per_atom_loss = -torch.log(softmax_max + 1e-8)
                    vec_losses.append(per_atom_loss[elig_mask])
                if vec_losses:
                    loss_mutual_exclusion = torch.cat(vec_losses).mean()

        if config.get("use_l1_sparsity", False):
            probs_full = torch.softmax(edit_logits_v25, dim=-1)
            loss_l1_sparsity = probs_full[:, 1:].sum(dim=-1).mean()

        if (
            config.get("use_edit_conditional_rc", False)
            and outputs.get("all_cand_global") is not None
            and outputs["rc_logits"].size(0) == batch.x.size(0)
        ):
            cand_global_v25 = outputs["all_cand_global"]
            nz = edit_labels_v25 != 0
            cond_label = torch.zeros(batch.x.size(0), dtype=torch.float, device=device)
            if nz.any():
                cond_label[cand_global_v25[nz, 0]] = 1.0
                cond_label[cand_global_v25[nz, 1]] = 1.0
            loss_edit_conditional_rc = F.binary_cross_entropy_with_logits(
                outputs["rc_logits"].squeeze(-1).float(), cond_label
            )

    loss_k_pred = torch.tensor(0.0, device=device)
    k_pred_acc = 0.0
    if (config.get("use_k_pred_head", False)
            and outputs.get("k_logits") is not None
            and outputs.get("rc_target_flat") is not None):
        wu_end = int(config.get("k_pred_warmup_epochs", 3))
        if epoch is None or int(epoch) >= wu_end:
            k_logits_b = outputs["k_logits"]
            batch_idx_kp = batch.batch
            rc_target_kp = outputs["rc_target_flat"]
            num_classes = int(config.get("k_pred_head_classes", 8))
            B_kp = k_logits_b.size(0)

            gt_K_per_batch = torch.zeros(B_kp, device=device, dtype=torch.long)
            ones_kp = torch.ones_like(rc_target_kp, dtype=torch.long)
            gt_K_per_batch.scatter_add_(0, batch_idx_kp, (rc_target_kp > 0).long())

            gt_K_per_batch = gt_K_per_batch.clamp(min=0, max=num_classes - 1)

            cw = config.get("k_pred_class_weights",
                            [1.0] * (num_classes - 3) + [3.0] * 3)
            cw_tensor = torch.tensor(cw, dtype=k_logits_b.dtype, device=device)
            loss_k_pred = F.cross_entropy(
                k_logits_b.float(), gt_K_per_batch, weight=cw_tensor.float()
            )

            with torch.no_grad():
                k_pred_acc = (
                    k_logits_b.argmax(dim=-1) == gt_K_per_batch
                ).float().mean().item()

    loss_k_aware_sparsity = torch.tensor(0.0, device=device)
    if config.get("use_k_aware_sparsity_loss", False) and outputs.get("rc_target_flat") is not None:
        wu_end = int(config.get("k_aware_warmup_epochs", 3))
        if epoch is None or int(epoch) >= wu_end:
            rc_probs_k = outputs["rc_probs"]
            batch_idx_k = batch.batch
            rc_target_k = outputs["rc_target_flat"].to(rc_probs_k.dtype)
            target_pad = int(config.get("k_aware_target_pad", 1))
            B_k = int(batch_idx_k.max().item()) + 1 if batch_idx_k.numel() > 0 else 0

            gt_K_per_batch = torch.zeros(B_k, device=device, dtype=rc_probs_k.dtype)
            if B_k > 0:
                gt_K_per_batch.scatter_add_(0, batch_idx_k, rc_target_k)

            push_losses = []
            for b in range(B_k):
                mask_b = batch_idx_k == b
                probs_b = rc_probs_k[mask_b]
                n_b = probs_b.numel()
                K_gt = int(gt_K_per_batch[b].item())
                allowed_K = K_gt + target_pad
                if n_b <= allowed_K or allowed_K < 0:
                    continue
                sorted_probs, _ = torch.sort(probs_b, descending=True)

                threshold = sorted_probs[allowed_K]
                push_mask = (probs_b <= threshold) & (probs_b > 0.1)
                if push_mask.any():
                    push_losses.append((probs_b[push_mask].float() ** 2).mean())
            if push_losses:
                loss_k_aware_sparsity = torch.stack(push_losses).mean()

    total = (
        config["w_rc"] * loss_rc
        + config["w_edit"] * edit_ramp * loss_edit
        + config.get("w_pair_sym", 0.0) * loss_pair_sym
        + config.get("w_bond_rc", 0.0) * loss_bond_rc
        + config.get("w_pair_contrastive", 0.0) * loss_pair_contrastive
        + config.get("w_pair_head", 0.0) * loss_pair_head
        + config.get("w_pair_teacher", 0.0) * loss_pair_teach
        + config.get("w_all_atom_hit", 0.0) * loss_all_atom_hit
        + config.get("w_pair_hard", 0.0) * loss_pair_hard
        + config.get("w_valence_rc", 0.0) * loss_valence_rc
        + config.get("w_pair_exist", 0.0) * loss_pair_exist
        + config.get("w_fukui_rank", 0.0) * loss_fukui_rank
        + config.get("w_mol_aux", 0.0) * loss_mol_aux
        + config.get("w_cc_pair_constraint", 0.0) * loss_cc_pair
        + config.get("w_set_recall", 0.0) * loss_set_recall
        + config.get("w_set_precision", 0.0) * loss_set_precision
        + get_w_pointer_rc_v22(
            epoch,
            base=float(config.get("w_pointer_rc", 0.0)),
            warmup_end=int(config.get("pointer_rc_warmup_end", 5)),
            decay_end=int(config.get("pointer_rc_decay_end", 8)),
        ) * loss_pointer_rc
        + config.get("w_valence_dh", 0.0) * loss_valence_dh

        + (config.get("w_margin_penalty", 0.0)
            if config.get("use_margin_penalty", False) else 0.0) * loss_margin_penalty
        + (config.get("w_mutual_exclusion", 0.0)
            if config.get("use_mutual_exclusion", False) else 0.0) * loss_mutual_exclusion
        + (config.get("w_edit_conditional_rc", 0.0)
            if config.get("use_edit_conditional_rc", False) else 0.0) * loss_edit_conditional_rc
        + (config.get("w_l1_sparsity", 0.0)
            if config.get("use_l1_sparsity", False) else 0.0) * loss_l1_sparsity

        + (config.get("w_k_aware_sparsity", 0.0)
            if config.get("use_k_aware_sparsity_loss", False) else 0.0) * loss_k_aware_sparsity

        + (config.get("w_k_pred", 0.0)
            if config.get("use_k_pred_head", False) else 0.0) * loss_k_pred
    )

    loss_rc_set_pred = torch.tensor(0.0, device=device)
    set_pred_metrics = {}
    if (config.get("use_rc_set_pred_head", False)
            and outputs.get("presence_logits") is not None
            and outputs.get("atom_logits_set") is not None):
        try:
            _set_out = compute_set_prediction_loss(
                outputs["presence_logits"],
                outputs["atom_logits_set"],
                outputs["mask_dense_set"],
                rc_target_flat if rc_target_flat is not None else torch.zeros(outputs["h_rc"].size(0), device=device),
                batch_idx,
                outputs["reactant_mask_v33j"],
                outputs["atom_idx_dense_set"],
                lambda_no=float(config.get("rc_set_pred_lambda_no", 0.5)),
                alpha_cost=float(config.get("rc_set_pred_alpha_cost", 1.0)),
            )
            loss_rc_set_pred = _set_out["loss_set"]
            set_pred_metrics = {
                "l_rc_set_pred": float(_set_out["loss_set"].detach().item()),
                "l_rc_set_atom_ce": float(_set_out["loss_atom_ce"].item()),
                "l_rc_set_presence": float(_set_out["loss_presence"].item()),
                "rc_set_matched": int(_set_out["matched_count"].item()),
                "rc_set_no_object": int(_set_out["no_object_count"].item()),
            }
        except Exception as _e:
            print(f"[v33_J] set_pred_loss failed: {_e}")
            loss_rc_set_pred = torch.tensor(0.0, device=device)
            set_pred_metrics = {}
    total = total + config.get("w_rc_set_pred", 0.0) * loss_rc_set_pred

    pair_cons_alpha = 0.0
    if model is not None:
        raw_model = getattr(model, "module", model)
        if getattr(raw_model, "pair_cons_gate", None) is not None:
            pair_cons_alpha = float(raw_model.pair_cons_gate.alpha.detach().float().item())

    _loss_vals = torch.stack([
        loss_rc, loss_edit,
        loss_pair_sym, loss_bond_rc, loss_pair_contrastive,
        loss_pair_head, loss_pair_teach,
        loss_all_atom_hit, loss_pair_hard,
        loss_valence_rc, loss_pair_exist, loss_fukui_rank,
        loss_mol_aux,
        loss_cc_pair,
        loss_set_recall, loss_set_precision,
        loss_pointer_rc,
        loss_valence_dh,
        loss_margin_penalty,
        loss_mutual_exclusion,
        loss_edit_conditional_rc,
        loss_l1_sparsity,
        loss_k_aware_sparsity,
        loss_k_pred,
    ]).detach().float().cpu()
    return total, {
        "l_rc": _loss_vals[0].item(),
        "l_edit": _loss_vals[1].item(),
        "l_pair_sym": _loss_vals[2].item(),
        "l_bond_rc": _loss_vals[3].item(),
        "l_pair_contrastive": _loss_vals[4].item(),
        "l_pair_head": _loss_vals[5].item(),
        "l_pair_teach": _loss_vals[6].item(),
        "l_all_atom_hit": _loss_vals[7].item(),
        "l_pair_hard": _loss_vals[8].item(),
        "l_valence_rc": _loss_vals[9].item(),
        "l_pair_exist": _loss_vals[10].item(),
        "l_fukui_rank": _loss_vals[11].item(),
        "l_mol_aux": _loss_vals[12].item(),
        "l_cc_pair": _loss_vals[13].item(),
        "l_set_recall": _loss_vals[14].item(),
        "l_set_precision": _loss_vals[15].item(),
        "l_pointer_rc": _loss_vals[16].item(),
        "l_valence_dh": _loss_vals[17].item(),
        "l_margin_penalty": _loss_vals[18].item(),
        "l_mutual_exclusion": _loss_vals[19].item(),
        "l_edit_cond_rc": _loss_vals[20].item(),
        "l_l1_sparsity": _loss_vals[21].item(),
        "l_k_aware_sparsity": _loss_vals[22].item(),
        "l_k_pred": _loss_vals[23].item(),
        "k_pred_acc": k_pred_acc,
        "pair_cons_alpha": pair_cons_alpha,

        **set_pred_metrics,
    }

def compute_metrics(outputs, batch, config):
    """计算指标: RC/Edit/Mol 三级诊断指标"""
    device = outputs["rc_logits"].device
    metrics = {}

    batch_idx = batch.batch
    batch_size = batch_idx.max().item() + 1

    rc_target_flat = outputs["rc_target_flat"]

    rc_probs = outputs["rc_probs"]
    rc_pred = (rc_probs > 0.5).float()

    pos_mask = rc_target_flat == 1
    neg_mask = rc_target_flat == 0
    _rc_stats = torch.stack([
        ((rc_pred == 1) & (rc_target_flat == 1)).sum().float(),
        (rc_target_flat == 1).sum().float(),
        (rc_pred == 1).sum().float(),
        rc_probs[pos_mask].sum() if pos_mask.any() else torch.tensor(0.0, device=device),
        pos_mask.sum().float(),
        rc_probs[neg_mask].sum() if neg_mask.any() else torch.tensor(0.0, device=device),
        neg_mask.sum().float(),
    ]).cpu()
    true_pos = _rc_stats[0].item()
    actual_pos = _rc_stats[1].item()
    pred_pos = _rc_stats[2].item()

    metrics["rc_recall"] = true_pos / max(actual_pos, 1)
    metrics["rc_precision"] = true_pos / max(pred_pos, 1)
    metrics["rc_f1"] = (
        2
        * metrics["rc_recall"]
        * metrics["rc_precision"]
        / max(metrics["rc_recall"] + metrics["rc_precision"], 1e-8)
    )

    metrics["rc_avg_prob_pos"] = (
        _rc_stats[3].item() / max(_rc_stats[4].item(), 1)
    )
    metrics["rc_avg_prob_neg"] = (
        _rc_stats[5].item() / max(_rc_stats[6].item(), 1)
    )
    metrics["rc_pred_count"] = pred_pos / max(batch_size, 1)
    metrics["rc_gt_count"] = actual_pos / max(batch_size, 1)

    if outputs.get("rc_pair_index") is not None and outputs["rc_pair_index"].numel() > 0:
        pair_idx = outputs["rc_pair_index"]
        rc_pred_long = rc_pred.long()
        i_hit = rc_pred_long[pair_idx[:, 0]]
        j_hit = rc_pred_long[pair_idx[:, 1]]
        both_hit = ((i_hit == 1) & (j_hit == 1)).float().mean().item()
        metrics["rc_both_rate"] = both_hit
    else:
        metrics["rc_both_rate"] = 0.0

    rc_tgt_flat_m = outputs.get("rc_target_flat")
    if rc_tgt_flat_m is not None:
        gt_mask_m = rc_tgt_flat_m.bool()
        if gt_mask_m.any():
            device_m = rc_pred.device
            rc_pred_long_m = rc_pred.long()
            miss_flag = (rc_pred_long_m[gt_mask_m] == 0).long()
            ones_flag = torch.ones_like(miss_flag)
            per_b_miss = torch.zeros(batch_size, device=device_m, dtype=torch.long)
            per_b_gt = torch.zeros(batch_size, device=device_m, dtype=torch.long)
            per_b_miss.index_add_(0, batch_idx[gt_mask_m], miss_flag)
            per_b_gt.index_add_(0, batch_idx[gt_mask_m], ones_flag)
            valid_b = per_b_gt > 0
            all_hit_b = (per_b_miss == 0) & valid_b
            metrics["all_atom_hit_rate"] = (
                all_hit_b.sum().item() / max(valid_b.sum().item(), 1)
            )
        else:
            metrics["all_atom_hit_rate"] = 0.0
    else:
        metrics["all_atom_hit_rate"] = 0.0

    if outputs["edit_logits"].size(0) > 0:
        edit_logits = outputs["edit_logits"]
        edit_pred = edit_logits.argmax(dim=-1)
        edit_true = outputs["edit_labels"]
        edit_probs = torch.softmax(edit_logits.float(), dim=-1)

        correct = (edit_pred == edit_true).sum().item()
        total = edit_true.size(0)
        metrics["edit_acc"] = correct / max(total, 1)

        true_nonzero = edit_true != 0
        pred_nonzero = edit_pred != 0

        tp_nonzero = ((edit_pred == edit_true) & true_nonzero).sum().item()
        metrics["edit_recall"] = tp_nonzero / max(true_nonzero.sum().item(), 1)
        metrics["edit_precision"] = tp_nonzero / max(pred_nonzero.sum().item(), 1)

        max_probs = edit_probs.max(dim=-1).values
        correct_mask = edit_pred == edit_true
        metrics["edit_conf_correct"] = (
            max_probs[correct_mask].mean().item() if correct_mask.any() else 0.0
        )
        metrics["edit_conf_wrong"] = (
            max_probs[~correct_mask].mean().item() if (~correct_mask).any() else 0.0
        )

        metrics["edit_nonzero_pred_count"] = pred_nonzero.sum().item() / max(
            batch_size, 1
        )
        metrics["edit_nonzero_gt_count"] = true_nonzero.sum().item() / max(
            batch_size, 1
        )

        metrics["cand_nonzero_ratio"] = true_nonzero.sum().item() / max(total, 1)

        per_class_acc = []
        for c in range(config["delta_classes"]):
            c_mask = edit_true == c
            if c_mask.sum().item() > 0:
                c_acc = (edit_pred[c_mask] == c).float().mean().item()
            else:
                c_acc = -1.0
            per_class_acc.append(f"{c_acc:.4f}")
        metrics["edit_per_class_acc"] = ";".join(per_class_acc)
    else:
        metrics["edit_acc"] = 0.0
        metrics["edit_recall"] = 0.0
        metrics["edit_precision"] = 0.0
        metrics["edit_conf_correct"] = 0.0
        metrics["edit_conf_wrong"] = 0.0
        metrics["edit_nonzero_pred_count"] = 0.0
        metrics["edit_nonzero_gt_count"] = 0.0
        metrics["cand_nonzero_ratio"] = 0.0
        metrics["edit_per_class_acc"] = ""

    mol_result = compute_mol_acc_detailed(outputs, batch, config)
    metrics["mol_acc"] = mol_result["mol_acc"]
    metrics["mol_fail_rc"] = mol_result["fail_rc"]
    metrics["mol_fail_edit"] = mol_result["fail_edit"]
    metrics["mol_fail_coverage"] = mol_result["fail_coverage"]
    metrics["cand_gt_coverage"] = mol_result["gt_coverage"]

    metrics["K0_mol_acc"] = mol_result.get("K0_mol_acc", 0.0)
    metrics["K2_mol_acc"] = mol_result.get("K2_mol_acc", 0.0)
    metrics["K3_mol_acc"] = mol_result.get("K3_mol_acc", 0.0)
    metrics["K4_mol_acc"] = mol_result.get("K4_mol_acc", 0.0)
    metrics["K5p_mol_acc"] = mol_result.get("K5p_mol_acc", 0.0)
    metrics["K0_count"] = mol_result.get("K0_count", 0.0)
    metrics["K2_count"] = mol_result.get("K2_count", 0.0)
    metrics["K3_count"] = mol_result.get("K3_count", 0.0)
    metrics["K4_count"] = mol_result.get("K4_count", 0.0)
    metrics["K5p_count"] = mol_result.get("K5p_count", 0.0)
    metrics["main_mol_acc"] = mol_result.get("main_mol_acc", 0.0)

    metrics["main_mol_acc_revised"] = mol_result.get("main_mol_acc_revised", 0.0)
    metrics["K_pos_main_mol_acc"] = mol_result.get("K_pos_main_mol_acc", 0.0)
    metrics["K_pos_count"] = mol_result.get("K_pos_count", 0.0)
    metrics["K4plus_main_mol_acc"] = mol_result.get("K4plus_main_mol_acc", 0.0)
    metrics["K4plus_count"] = mol_result.get("K4plus_count", 0.0)
    metrics["bucket_avg_acc"] = mol_result.get("bucket_avg_acc", 0.0)
    metrics["fail_rc_total"] = mol_result.get("fail_rc_total", 0.0)
    metrics["K0_main_revised"] = mol_result.get("K0_main_revised", 0.0)
    metrics["K2_main_revised"] = mol_result.get("K2_main_revised", 0.0)
    metrics["K3_main_revised"] = mol_result.get("K3_main_revised", 0.0)
    metrics["K4_main_revised"] = mol_result.get("K4_main_revised", 0.0)
    metrics["K5p_main_revised"] = mol_result.get("K5p_main_revised", 0.0)

    metrics["main_mol_acc_product"] = mol_result.get("main_mol_acc_product", 0.0)
    metrics["K_pos_product_mol_acc"] = mol_result.get("K_pos_product_mol_acc", 0.0)
    metrics["K0_product"] = mol_result.get("K0_product", 0.0)
    metrics["K2_product"] = mol_result.get("K2_product", 0.0)
    metrics["K3_product"] = mol_result.get("K3_product", 0.0)
    metrics["K4_product"] = mol_result.get("K4_product", 0.0)
    metrics["K5p_product"] = mol_result.get("K5p_product", 0.0)

    metrics["avg_cand_edges"] = outputs["num_cand_edges"] / max(batch_size, 1)

    return metrics

_RD_ORDER_TO_BT = None

def _v26_product_canonical(smi_r, changes, kept_atoms=None):
    """Apply bond delta changes to reactant, restrict to kept atoms, return canonical SMILES.

    changes: list of (i, j, delta_int)
    kept_atoms: set of reactant atom indices to keep (default = all atoms with atom_map > 0)
    """
    global _RD_ORDER_TO_BT
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    if _RD_ORDER_TO_BT is None:
        _RD_ORDER_TO_BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}

    r = Chem.MolFromSmiles(smi_r)
    if r is None: return ""
    try: Chem.Kekulize(r, clearAromaticFlags=True)
    except Exception: pass
    if kept_atoms is None or not kept_atoms:
        kept_atoms = {a.GetIdx() for a in r.GetAtoms() if a.GetAtomMapNum() > 0}
        if not kept_atoms:
            try:
                frags = Chem.GetMolFrags(r, asMols=False)
                if frags: kept_atoms = set(max(frags, key=len))
            except Exception: return ""
    keep_sorted = sorted(kept_atoms)
    o2n = {old: new for new, old in enumerate(keep_sorted)}
    sub = Chem.RWMol()
    for old_i in keep_sorted:
        ra = r.GetAtomWithIdx(old_i)
        na = Chem.Atom(ra.GetAtomicNum())
        na.SetFormalCharge(ra.GetFormalCharge())
        na.SetAtomMapNum(0)
        sub.AddAtom(na)
    cur_bond = {}
    for b in r.GetBonds():
        oi, oj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if oi not in o2n or oj not in o2n: continue
        t = b.GetBondType()
        order = (1 if t == Chem.BondType.SINGLE else 2 if t == Chem.BondType.DOUBLE else 3 if t == Chem.BondType.TRIPLE else 1)
        sub.AddBond(o2n[oi], o2n[oj], _RD_ORDER_TO_BT[order])
        cur_bond[(min(o2n[oi], o2n[oj]), max(o2n[oi], o2n[oj]))] = order
    for old_i, old_j, delta in changes:
        if old_i not in o2n or old_j not in o2n: continue
        if delta == 0: continue
        ni, nj = o2n[old_i], o2n[old_j]
        k = (min(ni, nj), max(ni, nj))
        cur = cur_bond.get(k, 0)
        new_o = max(0, min(3, cur + delta))
        b_ex = sub.GetBondBetweenAtoms(ni, nj)
        if b_ex is not None: sub.RemoveBond(ni, nj)
        if new_o > 0: sub.AddBond(ni, nj, _RD_ORDER_TO_BT[new_o])
        cur_bond[k] = new_o
    try:
        Chem.SanitizeMol(sub)
        return Chem.MolToSmiles(sub, canonical=True)
    except Exception:
        try: return Chem.MolToSmiles(sub, canonical=True)
        except Exception: return ""

_CLS_TO_DELTA_INT = {0: 0, 1: +1, 2: -1, 3: +2, 4: -2, 5: +3, 6: -3}

def v26_product_match_for_mol(smi_r, smi_p, cand_local, mol_pred_cls, mol_true_cls):
    """Compute product-level match for one molecule. Returns True if pred & GT make same product."""
    if not smi_r or not smi_p:
        return None
    from rdkit import Chem
    r = Chem.MolFromSmiles(smi_r)
    p = Chem.MolFromSmiles(smi_p)
    if r is None or p is None:
        return None
    p_maps = {a.GetAtomMapNum() for a in p.GetAtoms() if a.GetAtomMapNum() > 0}
    kept_atoms = {a.GetIdx() for a in r.GetAtoms() if a.GetAtomMapNum() > 0 and a.GetAtomMapNum() in p_maps}
    if not kept_atoms:
        return None

    pred_changes = []
    gt_changes = []
    for k in range(cand_local.size(0)):
        i = int(cand_local[k, 0].item())
        j = int(cand_local[k, 1].item())
        pc = int(mol_pred_cls[k].item())
        gc = int(mol_true_cls[k].item())
        if pc != 0:
            pred_changes.append((i, j, _CLS_TO_DELTA_INT.get(pc, 0)))
        if gc != 0:
            gt_changes.append((i, j, _CLS_TO_DELTA_INT.get(gc, 0)))
    pred_canon = _v26_product_canonical(smi_r, pred_changes, kept_atoms)
    gt_canon = _v26_product_canonical(smi_r, gt_changes, kept_atoms)
    if not pred_canon or not gt_canon:
        return None
    return pred_canon == gt_canon

def compute_mol_acc_detailed(outputs, batch, config):
    """分子级准确率 + 失败原因拆分.

    v20.4: 扩展 per-K mol_acc (K=0/2/3/4/5+) + 主产物 mol_acc (largest CC).
    """
    result = {
        "mol_acc": 0.0,
        "main_mol_acc_revised": 0.0,
        "K_pos_main_mol_acc": 0.0,
        "K_pos_count": 0.0,
        "K4plus_main_mol_acc": 0.0,
        "K4plus_count": 0.0,
        "bucket_avg_acc": 0.0,
        "fail_rc_total": 0.0,
        "K0_main_revised": 0.0,
        "K2_main_revised": 0.0,
        "K3_main_revised": 0.0,
        "K4_main_revised": 0.0,
        "K5p_main_revised": 0.0,
        "fail_rc": 0.0,
        "fail_edit": 0.0,
        "fail_coverage": 0.0,
        "gt_coverage": 1.0,

        "K0_mol_acc": 0.0, "K0_count": 0.0,
        "K2_mol_acc": 0.0, "K2_count": 0.0,
        "K3_mol_acc": 0.0, "K3_count": 0.0,
        "K4_mol_acc": 0.0, "K4_count": 0.0,
        "K5p_mol_acc": 0.0, "K5p_count": 0.0,

        "main_mol_acc": 0.0,
        "main_mol_acc_product": 0.0,
        "K_pos_product_mol_acc": 0.0,
        "K0_product": 0.0, "K2_product": 0.0, "K3_product": 0.0, "K4_product": 0.0, "K5p_product": 0.0,
    }

    if outputs["edit_logits"].size(0) == 0:
        return result

    device = outputs["edit_logits"].device
    batch_idx = batch.batch
    batch_size = batch_idx.max().item() + 1
    y_delta_list = batch.y_delta_list

    edit_pred = outputs["edit_logits"].argmax(dim=-1)
    edit_labels = outputs["edit_labels"]
    rc_probs = outputs["rc_probs"]

    edge_counts = [e.size(0) for e in outputs["cand_edges"]]
    cand_edges_list = outputs.get("cand_edges", [])
    node_counts = torch.bincount(batch_idx, minlength=batch_size)
    node_offsets = torch.cat(
        [torch.tensor([0], device=device), node_counts.cumsum(0)[:-1]]
    )

    correct_mols = 0
    fail_rc = 0
    fail_edit = 0
    fail_coverage = 0
    total_gt_edits = 0
    total_covered_edits = 0
    offset = 0

    K_buckets = {0: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
    main_correct = 0

    K_buckets_main_revised = {0: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
    main_correct_revised = 0

    K_buckets_product = {0: [0, 0], 2: [0, 0], 3: [0, 0], 4: [0, 0], 5: [0, 0]}
    main_correct_product = 0
    smi_r_list = getattr(batch, "smi_r_list", None)
    smi_p_list = getattr(batch, "smi_p_list", None)
    use_product_eval = config.get("use_product_level_eval", True) and smi_r_list is not None and smi_p_list is not None

    use_main = config.get("use_main_product_judgment", False)

    for mol_idx in range(batch_size):
        k = edge_counts[mol_idx]
        y_delta = y_delta_list[mol_idx].to(device)
        gt_nonzero = (y_delta != 0).triu(diagonal=1)
        gt_edit_count = gt_nonzero.sum().item()

        rc_target_local = extract_rc_target(y_delta)
        K = int((rc_target_local > 0.5).sum().item())
        K_bucket = 0 if K == 0 else (2 if K == 2 else (3 if K == 3 else (4 if K == 4 else 5)))
        K_buckets[K_bucket][1] += 1

        if use_main:
            n_local = int(node_counts[mol_idx].item())
            start_local = int(node_offsets[mol_idx].item())
            edge_mask_local = (
                (batch.edge_index[0] >= start_local)
                & (batch.edge_index[0] < start_local + n_local)
            )
            local_ei = (batch.edge_index[:, edge_mask_local] - start_local).cpu()
            local_ea = batch.edge_attr[edge_mask_local].cpu().float()
            main_mask_local = compute_main_product_atoms(
                local_ei, local_ea, y_delta.cpu(), n_local,
            )
        else:
            main_mask_local = None

        if k == 0:
            if gt_edit_count == 0:
                correct_mols += 1
                K_buckets[K_bucket][0] += 1
                main_correct += 1

                K_buckets_main_revised[K_bucket][0] += 1
                main_correct_revised += 1
            else:
                fail_coverage += 1
                total_gt_edits += gt_edit_count
            continue

        mol_pred = edit_pred[offset : offset + k]
        mol_true = edit_labels[offset : offset + k]
        cand_for_mol = cand_edges_list[mol_idx] if mol_idx < len(cand_edges_list) else None
        offset += k

        covered_edits = (mol_true != 0).sum().item()
        total_gt_edits += gt_edit_count
        total_covered_edits += covered_edits

        if gt_edit_count != covered_edits:
            fail_coverage += 1
            continue

        if not (mol_pred == mol_true).all():
            start = node_offsets[mol_idx].item()
            n = node_counts[mol_idx].item()
            mol_rc_probs = rc_probs[start : start + n]
            mol_rc_pred = mol_rc_probs > 0.5
            rc_correct = (mol_rc_pred == (rc_target_local > 0.5)).all().item()
            if not rc_correct:
                fail_rc += 1
            else:
                fail_edit += 1

            if use_main and main_mask_local is not None and cand_for_mol is not None:
                cand_local = cand_for_mol.cpu()
                mask_main_edge = main_mask_local[cand_local[:, 0]] | main_mask_local[cand_local[:, 1]]

                if mask_main_edge.any():
                    pred_main_match = (mol_pred.cpu()[mask_main_edge] == mol_true.cpu()[mask_main_edge]).all().item()
                else:
                    pred_main_match = True
                if pred_main_match:
                    main_correct += 1

                gt_edge_mask_cpu = (mol_true.cpu() != 0)
                both_in_main = main_mask_local[cand_local[:, 0]] & main_mask_local[cand_local[:, 1]]
                mask_main_revised = both_in_main | gt_edge_mask_cpu
                if mask_main_revised.any():
                    pred_main_match_revised = (mol_pred.cpu()[mask_main_revised] == mol_true.cpu()[mask_main_revised]).all().item()
                else:
                    pred_main_match_revised = True
                if pred_main_match_revised:
                    main_correct_revised += 1
                    K_buckets_main_revised[K_bucket][0] += 1

                if use_product_eval and not pred_main_match_revised and K > 0:
                    try:
                        pl = v26_product_match_for_mol(
                            smi_r_list[mol_idx], smi_p_list[mol_idx],
                            cand_local, mol_pred.cpu(), mol_true.cpu(),
                        )
                    except Exception:
                        pl = None
                    if pl is True:
                        main_correct_product += 1
                        K_buckets_product[K_bucket][0] += 1
                elif pred_main_match_revised:
                    main_correct_product += 1
                    K_buckets_product[K_bucket][0] += 1
            continue

        correct_mols += 1
        K_buckets[K_bucket][0] += 1
        main_correct += 1

        K_buckets_main_revised[K_bucket][0] += 1
        main_correct_revised += 1

        K_buckets_product[K_bucket][0] += 1
        main_correct_product += 1

    result["mol_acc"] = correct_mols / max(batch_size, 1)
    result["fail_rc"] = fail_rc / max(batch_size, 1)
    result["fail_edit"] = fail_edit / max(batch_size, 1)
    result["fail_coverage"] = fail_coverage / max(batch_size, 1)

    result["fail_rc_total"] = (fail_rc + fail_coverage) / max(batch_size, 1)
    result["gt_coverage"] = total_covered_edits / max(total_gt_edits, 1)

    result["K0_mol_acc"] = K_buckets[0][0] / max(K_buckets[0][1], 1)
    result["K0_count"] = K_buckets[0][1]
    result["K2_mol_acc"] = K_buckets[2][0] / max(K_buckets[2][1], 1)
    result["K2_count"] = K_buckets[2][1]
    result["K3_mol_acc"] = K_buckets[3][0] / max(K_buckets[3][1], 1)
    result["K3_count"] = K_buckets[3][1]
    result["K4_mol_acc"] = K_buckets[4][0] / max(K_buckets[4][1], 1)
    result["K4_count"] = K_buckets[4][1]
    result["K5p_mol_acc"] = K_buckets[5][0] / max(K_buckets[5][1], 1)
    result["K5p_count"] = K_buckets[5][1]
    result["main_mol_acc"] = main_correct / max(batch_size, 1)

    result["main_mol_acc_revised"] = main_correct_revised / max(batch_size, 1)

    K_pos_correct = main_correct_revised - K_buckets_main_revised[0][0]
    K_pos_total = batch_size - K_buckets[0][1]
    result["K_pos_main_mol_acc"] = K_pos_correct / max(K_pos_total, 1)

    result["main_mol_acc_product"] = main_correct_product / max(batch_size, 1)
    K_pos_correct_product = main_correct_product - K_buckets_product[0][0]
    result["K_pos_product_mol_acc"] = K_pos_correct_product / max(K_pos_total, 1)
    for k_val, label in [(0, "K0"), (2, "K2"), (3, "K3"), (4, "K4"), (5, "K5p")]:
        result[f"{label}_product"] = K_buckets_product[k_val][0] / max(K_buckets[k_val][1], 1)
    result["K_pos_count"] = K_pos_total

    K4plus_correct = K_buckets_main_revised[4][0] + K_buckets_main_revised[5][0]
    K4plus_total = K_buckets[4][1] + K_buckets[5][1]
    result["K4plus_main_mol_acc"] = K4plus_correct / max(K4plus_total, 1)
    result["K4plus_count"] = K4plus_total

    bucket_accs = []
    for k_val in [2, 3, 4, 5]:
        if K_buckets[k_val][1] > 0:
            bucket_accs.append(K_buckets_main_revised[k_val][0] / K_buckets[k_val][1])
    result["bucket_avg_acc"] = sum(bucket_accs) / max(len(bucket_accs), 1)

    result["K0_main_revised"] = K_buckets_main_revised[0][0] / max(K_buckets[0][1], 1)
    result["K2_main_revised"] = K_buckets_main_revised[2][0] / max(K_buckets[2][1], 1)
    result["K3_main_revised"] = K_buckets_main_revised[3][0] / max(K_buckets[3][1], 1)
    result["K4_main_revised"] = K_buckets_main_revised[4][0] / max(K_buckets[4][1], 1)
    result["K5p_main_revised"] = K_buckets_main_revised[5][0] / max(K_buckets[5][1], 1)

    return result

def find_optimal_rc_threshold(model, val_loader, device):
    """
    Phase 1 结束后在验证集上搜索最优 RC 阈值。
    使用 coverage-aware 评分：score = 0.6*recall + 0.4*F1，优先保证高 recall（→高覆盖率）。
    同时评估每个阈值下的候选边 GT 覆盖率作为参考。
    """
    model.eval()
    rc_all_pairs = CONFIG.get("rc_all_pairs", False)
    rc_min_top_k = CONFIG.get("rc_min_top_k", 0)

    mol_data_list = []
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch = batch.to(device)
            outputs = model(batch, training=False, phase=1)
            all_probs.append(outputs["rc_probs"].cpu())

            batch_size = batch.batch.max().item() + 1
            batch_idx = batch.batch
            node_counts = torch.bincount(batch_idx, minlength=batch_size)
            node_offsets = torch.zeros(batch_size, dtype=torch.long, device=device)
            node_offsets[1:] = node_counts[:-1].cumsum(0)

            targets = []
            for mol_idx in range(batch_size):
                start = node_offsets[mol_idx].item()
                n = node_counts[mol_idx].item()
                y_delta = batch.y_delta_list[mol_idx]
                targets.append(extract_rc_target(y_delta))

                edge_mask = (batch.edge_index[0] >= start) & (
                    batch.edge_index[0] < start + n
                )
                mol_ei = (batch.edge_index[:, edge_mask] - start).cpu()
                mol_ea = batch.edge_attr[edge_mask].cpu()
                mol_data_list.append(
                    (
                        outputs["rc_probs"][start : start + n].cpu(),
                        y_delta,
                        mol_ei,
                        mol_ea,
                        n,
                    )
                )
            all_targets.append(torch.cat(targets).cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    best_score, best_thr = 0, 0.5
    print("  RC 阈值搜索 (coverage-aware):")
    for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        pred = (all_probs > thr).float()
        tp = ((pred == 1) & (all_targets == 1)).sum().item()
        fp = ((pred == 1) & (all_targets == 0)).sum().item()
        fn = ((pred == 0) & (all_targets == 1)).sum().item()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        total_gt, total_covered = 0, 0
        for mol_probs, y_delta, mol_ei, mol_ea, n in mol_data_list:
            rc_mask = mol_probs > thr
            if rc_min_top_k > 0 and rc_mask.sum() < rc_min_top_k:
                k = min(rc_min_top_k, n)
                _, topk_idx = mol_probs.topk(k)
                rc_mask = torch.zeros(n, dtype=torch.bool)
                rc_mask[topk_idx] = True
            cand_edges_np, _ = build_candidate_edges_cpu(
                mol_ei.numpy(), mol_ea.numpy(), n, rc_mask.numpy(),
                y_delta_np=None, rc_all_pairs=rc_all_pairs,
            )
            gt_nz = np.argwhere(np.triu(y_delta.cpu().numpy(), k=1) != 0)
            total_gt += len(gt_nz)
            if len(cand_edges_np) > 0 and len(gt_nz) > 0:
                cand_set = set(map(tuple, cand_edges_np))
                covered = sum(1 for r, c in gt_nz if (r, c) in cand_set)
                total_covered += covered

        coverage = total_covered / max(total_gt, 1)

        score = 0.6 * rec + 0.4 * f1
        print(
            f"    thr={thr:.2f}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} cov={coverage:.3f} score={score:.3f}"
        )
        if score > best_score:
            best_score, best_thr = score, thr

    print(f"  >>> 最优阈值={best_thr:.2f}, score={best_score:.3f}")
    return best_thr

def calibrate_temperature(model, val_loader, device):
    """v12: Phase 1 结束后校准温度，消除 ECE 偏差

    在 val set 上收集所有 rc_logits/labels，1D 搜索最优温度 T*，
    使得 sigmoid(logits / T*) 的 ECE 最小。Phase 2 推理使用校准后的温度。
    """
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch = batch.to(device)
            outputs = model(batch, training=False, phase=1)
            all_logits.append(outputs["rc_logits"].float().cpu())
            all_labels.append(outputs["rc_target_flat"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)

    def _ece(probs, labels, n_bins=15):
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        ece = 0.0
        total = len(probs)
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            avg_conf = probs[mask].mean().item()
            avg_acc = labels[mask].mean().item()
            ece += mask.sum().item() / total * abs(avg_conf - avg_acc)
        return ece

    best_t, best_ece = 1.0, float("inf")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0, 2.5, 3.0]:
        probs = torch.sigmoid(logits / t)
        ece = _ece(probs, labels)
        if ece < best_ece:
            best_t, best_ece = t, ece
    print(f"  >>> 温度校准: T*={best_t:.1f}, ECE={best_ece:.4f}")
    return best_t

def constrained_decode(edit_logits, cand_edges, cand_bond_types, atom_numbers=None):
    """
    约束解码 (推理时):
    1. 按置信度排序所有非零预测
    2. 贪心应用: 检查价态守恒
    3. 跳过违反化学约束的预测

    Args:
            edit_logits: [K, 7]
            cand_edges: [K, 2]
            cand_bond_types: [K] 当前键类型
            atom_numbers: [N] 原子序数 (可选, 用于价态查表)

    Returns:
            final_preds: [K] 最终预测类别
    """
    K = edit_logits.size(0)
    if K == 0:
        return torch.empty((0,), dtype=torch.long, device=edit_logits.device)

    device = edit_logits.device
    probs = F.softmax(edit_logits, dim=-1)

    pred_classes = probs.argmax(dim=-1)
    confidences = probs.max(dim=-1).values

    nonzero_mask = pred_classes != 0
    nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

    if nonzero_indices.size(0) == 0:
        return pred_classes

    nonzero_conf = confidences[nonzero_indices]
    sorted_order = nonzero_conf.argsort(descending=True)
    sorted_indices = nonzero_indices[sorted_order]

    num_atoms = max(cand_edges.max().item() + 1, 1)
    valence_change = torch.zeros(num_atoms, device=device)

    final_preds = torch.zeros(K, dtype=torch.long, device=device)

    for idx in sorted_indices:
        idx = idx.item()
        cls = pred_classes[idx].item()
        delta_val = CLASS_TO_DELTA[cls].item()

        i, j = cand_edges[idx, 0].item(), cand_edges[idx, 1].item()

        new_val_i = valence_change[i] + abs(delta_val)
        new_val_j = valence_change[j] + abs(delta_val)

        if new_val_i > 4 or new_val_j > 4:
            continue

        current_bond = cand_bond_types[idx].item()

        if current_bond == 4:
            effective_current = 1.5
        else:
            effective_current = float(current_bond)

        new_bond = effective_current + delta_val
        if new_bond < 0:
            continue

        final_preds[idx] = cls
        valence_change[i] += abs(delta_val)
        valence_change[j] += abs(delta_val)

    return final_preds

def main():

    rank, world_size, is_ddp = setup_ddp()
    device = torch.device(f"cuda:{rank}")
    CONFIG["device"] = device

    if is_main_process():
        os.makedirs(CONFIG["save_dir"], exist_ok=True)

        import shutil
        snapshot_path = os.path.join(
            CONFIG["save_dir"], os.path.basename(__file__) + ".snapshot"
        )
        if not os.path.exists(snapshot_path):
            shutil.copy2(__file__, snapshot_path)
            print(f"  训练脚本快照已保存: {snapshot_path}")
    if is_ddp:
        dist.barrier()

    if is_main_process():
        print(
            f"DDP: {world_size} GPU(s), batch/GPU={CONFIG['loader_batch_size']}, "
            f"accum={CONFIG['accum_steps']}, "
            f"effective_batch={CONFIG['loader_batch_size'] * world_size * CONFIG['accum_steps']}"
        )

    val_dir = CONFIG.get("val_dir", None)
    if val_dir and os.path.isdir(val_dir):
        if is_main_process():
            print(f"[v26 data] using explicit val_dir = {val_dir}")
        train_full = DiskDataset(
            CONFIG["data_dir"], augment=False, data_ratio=CONFIG["data_ratio"]
        )
        val_ds = DiskDataset(
            val_dir, augment=False, data_ratio=CONFIG["data_ratio"]
        )
        full_ds = train_full
        train_aug_ds = copy.copy(train_full)
        train_aug_ds.augment = True

        train_ds = torch.utils.data.Subset(train_aug_ds, list(range(len(train_full))))
    else:
        full_ds = DiskDataset(
            CONFIG["data_dir"], augment=False, data_ratio=CONFIG["data_ratio"]
        )

        generator = torch.Generator().manual_seed(42)
        train_size = int(0.9 * len(full_ds))
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [train_size, len(full_ds) - train_size], generator=generator
        )
        train_aug_ds = copy.copy(full_ds)
        train_aug_ds.augment = True
        train_ds = torch.utils.data.Subset(train_aug_ds, train_ds.indices)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if is_ddp else None

    if CONFIG.get("use_weighted_sampler", False) and train_sampler is None:

        cache_path = os.path.join(CONFIG["save_dir"], "chunk_weights_v15_1.pt")
        full_weights = precompute_chunk_weights(full_ds.files, cache_path=cache_path, verbose=is_main_process())

        try:
            train_indices = torch.tensor(list(train_ds.indices), dtype=torch.long)
        except Exception:

            train_indices = torch.tensor(list(range(len(train_ds))), dtype=torch.long)
        train_weights = full_weights[train_indices]
        from torch.utils.data import WeightedRandomSampler
        sampler_g = torch.Generator().manual_seed(CONFIG.get("sampler_seed", 42))
        train_sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_ds), replacement=True, generator=sampler_g
        )
        if is_main_process():
            print(f"[v15.1] WeightedRandomSampler active: num_samples={len(train_ds)}, "
                  f"mean_w={train_weights.mean().item():.3f}, "
                  f">=2 chunks: {int((train_weights>=2).sum())}/{len(train_weights)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["loader_batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=CONFIG["num_workers"],
        collate_fn=editgnn_collate,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=CONFIG.get("prefetch_factor", 2),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["loader_batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=CONFIG["num_workers"],
        collate_fn=editgnn_collate_eval,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=CONFIG.get("prefetch_factor", 2),
    )

    pretrained = load_pretrained_model(CONFIG["elec_ckpt_path"], device)
    model = EditGNNModel(pretrained, CONFIG).to(device)

    contrastive_loss_fn = ContrastiveRCLoss(
        temperature=CONFIG.get("contrastive_temp", 0.07),
        hidden_dim=CONFIG["hidden_dim"],
    ).to(device)

    if CONFIG.get("use_compile", False) and hasattr(torch, "compile"):
        if is_main_process():
            print("Applying torch.compile (default mode)...")

        model.edit_classifier = torch.compile(
            model.edit_classifier, mode="default"
        )

    raw_model = model
    if is_ddp:

        model = DDP(
            model,
            device_ids=[rank],
            find_unused_parameters=CONFIG.get("ddp_find_unused_parameters", True),
            gradient_as_bucket_view=CONFIG.get("ddp_gradient_as_bucket_view", False),
            broadcast_buffers=CONFIG.get("ddp_broadcast_buffers", True),
        )

    if is_main_process():
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(
            p.numel() for p in raw_model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Frozen params: {frozen_params:,}")

    bert_params = list(raw_model.text_encoder.bert.parameters())
    bert_param_ids = set(id(p) for p in bert_params)

    encoder_top_params = [
        p for p in raw_model.atom_model.parameters() if p.requires_grad
    ]
    encoder_top_param_ids = set(id(p) for p in encoder_top_params)

    frozen_param_ids = set(
        id(p) for p in raw_model.atom_model.parameters() if not p.requires_grad
    )

    dead_param_ids = set()
    if not CONFIG.get("rc_refine_enabled", True):
        dead_param_ids.update(id(p) for p in raw_model.rc_refiner.parameters())
        for p in raw_model.rc_refiner.parameters():
            p.requires_grad = False

    cross_frag_params = (
        list(raw_model.cross_frag_attn.parameters())
        + list(raw_model.frag_descriptor.parameters())
    )
    cross_frag_param_ids = set(id(p) for p in cross_frag_params)

    v16_new_params = []
    if getattr(raw_model.rc_head, "global_anchor", None) is not None:
        v16_new_params += list(raw_model.rc_head.global_anchor.parameters())
    if getattr(raw_model, "pair_sym_head", None) is not None:
        v16_new_params += list(raw_model.pair_sym_head.parameters())
    if getattr(raw_model, "pair_hard_head", None) is not None:
        v16_new_params += list(raw_model.pair_hard_head.parameters())

    if getattr(raw_model, "rc_set_refiner", None) is not None:
        v16_new_params += list(raw_model.rc_set_refiner.parameters())

    v16_new_param_ids = set(id(p) for p in v16_new_params)

    v19_pair_cons_params = []
    if getattr(raw_model, "pair_cons_gate", None) is not None:
        v19_pair_cons_params = list(raw_model.pair_cons_gate.parameters())
    v19_pair_cons_param_ids = set(id(p) for p in v19_pair_cons_params)

    v20_2_edit_inter_new_params = []
    n_edge_t_layers = CONFIG.get("n_edge_transformer_layers", 3)
    if n_edge_t_layers > 3 and hasattr(raw_model.edge_transformer, "transformer"):
        new_layer_idx = n_edge_t_layers - 1
        v20_2_edit_inter_new_params = list(
            raw_model.edge_transformer.transformer.layers[new_layer_idx].parameters()
        )
    v20_2_edit_inter_new_param_ids = set(id(p) for p in v20_2_edit_inter_new_params)

    v20_extra_params = []
    if getattr(raw_model, "rlc_film", None) is not None:
        v20_extra_params += list(raw_model.rlc_film.parameters())
    if getattr(raw_model, "pfrh", None) is not None:
        v20_extra_params += list(raw_model.pfrh.parameters())
    if getattr(raw_model, "edit_rlc_proj", None) is not None:
        v20_extra_params += list(raw_model.edit_rlc_proj.parameters())

    if getattr(raw_model, "role_signal_proj", None) is not None:
        v20_extra_params += list(raw_model.role_signal_proj.parameters())

    if getattr(raw_model, "multi_hop_rc_gate", None) is not None:
        v20_extra_params.append(raw_model.multi_hop_rc_gate)
    if getattr(raw_model, "multi_hop_rc_proj", None) is not None:
        v20_extra_params += list(raw_model.multi_hop_rc_proj.parameters())

    if getattr(raw_model, "k_pred_head", None) is not None:
        v20_extra_params += list(raw_model.k_pred_head.parameters())

    rc_param_ids = set(
        id(p)
        for p in (
            list(raw_model.adapter_rc.parameters())
            + list(raw_model.rc_head.parameters())
            + list(raw_model.edge_rc_head.parameters())
            + list(raw_model.rc_pair_refiner.parameters())
            + list(raw_model.env_encoder.parameters())
            + list(raw_model.text_encoder.projection.parameters())
            + list(raw_model.cond_encoder.parameters())
            + list(raw_model.global_project.parameters())
            + list(contrastive_loss_fn.parameters())
            + v20_extra_params
        )
    )

    rc_param_ids -= v16_new_param_ids

    v33j_param_ids = set()
    if getattr(raw_model, "rc_set_pred_head", None) is not None:
        v33j_param_ids = set(id(p) for p in raw_model.rc_set_pred_head.parameters())

    v34_e1_detr_edit_param_ids = set()
    if getattr(raw_model, "detr_edit_head", None) is not None:
        v34_e1_detr_edit_param_ids = set(
            id(p) for p in raw_model.detr_edit_head.parameters()
        )

    v34_e1_adapter_edit_param_ids = set()
    if CONFIG.get("use_detr_edit_head", False) and getattr(raw_model, "adapter_edit", None) is not None:
        v34_e1_adapter_edit_param_ids = set(
            id(p) for p in raw_model.adapter_edit.parameters()
        )

    rc_params = [
        p for p in raw_model.parameters()
        if p.requires_grad and id(p) in rc_param_ids and id(p) not in v16_new_param_ids
    ] + [p for p in contrastive_loss_fn.parameters() if p.requires_grad]
    edit_params = [
        p
        for p in raw_model.parameters()
        if p.requires_grad
        and id(p) not in bert_param_ids
        and id(p) not in frozen_param_ids
        and id(p) not in encoder_top_param_ids
        and id(p) not in rc_param_ids
        and id(p) not in cross_frag_param_ids
        and id(p) not in v16_new_param_ids
        and id(p) not in v19_pair_cons_param_ids
        and id(p) not in v20_2_edit_inter_new_param_ids
        and id(p) not in v33j_param_ids
        and id(p) not in v34_e1_detr_edit_param_ids
        and id(p) not in v34_e1_adapter_edit_param_ids
        and id(p) not in dead_param_ids
    ]

    v16_new_lr = CONFIG["lr_rc"] * CONFIG.get("new_module_lr_mult", 1.0)

    param_groups_list = [
        {
            "params": [p for p in bert_params if p.requires_grad],
            "lr": CONFIG["lr_bert"],
        },
        {"params": rc_params, "lr": CONFIG["lr_rc"]},
        {"params": edit_params, "lr": CONFIG["lr_edit"]},
        {"params": encoder_top_params, "lr": CONFIG.get("lr_encoder_top", 1e-5)},
        {"params": cross_frag_params, "lr": CONFIG.get("lr_cross_frag", 3e-4)},
    ]
    if v16_new_params:
        param_groups_list.append({"params": v16_new_params, "lr": v16_new_lr})
        if is_main_process():
            n_new_params = sum(p.numel() for p in v16_new_params)
            print(
                f"[v16/v18] Group 5 (GlobalAnchor + PairSymHead + PairHardHead): "
                f"{len(v16_new_params)} tensors, {n_new_params/1e6:.2f}M params, lr={v16_new_lr:.2e}"
            )

    if v19_pair_cons_params:
        v19_pc_lr = CONFIG.get("lr_pair_cons", CONFIG["lr_rc"])
        param_groups_list.append({"params": v19_pair_cons_params, "lr": v19_pc_lr})
        if is_main_process():
            print(
                f"[v19] Group 6 (PairConsistencyGate.alpha): "
                f"{len(v19_pair_cons_params)} tensors, "
                f"{sum(p.numel() for p in v19_pair_cons_params)} params, lr={v19_pc_lr:.2e}"
            )

    if v20_2_edit_inter_new_params:
        v20_2_eit_lr = CONFIG.get("lr_edit_inter_new", CONFIG["lr_edit"] * 3.0)
        param_groups_list.append(
            {"params": v20_2_edit_inter_new_params, "lr": v20_2_eit_lr}
        )
        if is_main_process():
            n_eit = sum(p.numel() for p in v20_2_edit_inter_new_params)
            print(
                f"[v20.2] Group 7 (EdgeT[{n_edge_t_layers - 1}] new layer): "
                f"{len(v20_2_edit_inter_new_params)} tensors, "
                f"{n_eit/1e6:.2f}M params, lr={v20_2_eit_lr:.2e}"
            )

    if getattr(raw_model, "rc_set_pred_head", None) is not None:
        v33j_params = list(raw_model.rc_set_pred_head.parameters())
        v33j_lr = float(CONFIG.get("rc_set_pred_lr", 5e-4))
        param_groups_list.append({"params": v33j_params, "lr": v33j_lr})
        if is_main_process():
            n_v33j = sum(p.numel() for p in v33j_params)
            print(
                f"[v33_J] Group 8 (RCSetPredHead DETR-style, fresh init): "
                f"{len(v33j_params)} tensors, {n_v33j/1e6:.2f}M params, lr={v33j_lr:.2e}"
            )

    v34_e1_detr_edit_group_idx = None
    if getattr(raw_model, "detr_edit_head", None) is not None:
        v34_e1_detr_edit_params = list(raw_model.detr_edit_head.parameters())
        v34_e1_detr_edit_lr_phase0 = float(CONFIG.get("lr_detr_edit_phase0", 5e-4))
        param_groups_list.append(
            {"params": v34_e1_detr_edit_params, "lr": v34_e1_detr_edit_lr_phase0}
        )
        v34_e1_detr_edit_group_idx = len(param_groups_list) - 1
        if is_main_process():
            n_v34e1 = sum(p.numel() for p in v34_e1_detr_edit_params)
            print(
                f"[v34_E1] Group {v34_e1_detr_edit_group_idx} "
                f"(DETREditHead, fresh init): "
                f"{len(v34_e1_detr_edit_params)} tensors, "
                f"{n_v34e1/1e6:.2f}M params, "
                f"lr={v34_e1_detr_edit_lr_phase0:.2e} (Phase 0); "
                f"will switch to {CONFIG.get('lr_detr_edit_phase1', 1e-4):.2e} "
                f"at epoch {CONFIG.get('phase1_start_epoch', 6)}"
            )

    v34_e1_adapter_edit_group_idx = None
    if v34_e1_adapter_edit_param_ids:
        v34_e1_adapter_edit_params = [
            p for p in raw_model.adapter_edit.parameters() if p.requires_grad
        ]
        v34_e1_adapter_edit_lr_phase0 = float(CONFIG.get("lr_adapter_edit_phase0", 0.0))
        param_groups_list.append(
            {"params": v34_e1_adapter_edit_params, "lr": v34_e1_adapter_edit_lr_phase0}
        )
        v34_e1_adapter_edit_group_idx = len(param_groups_list) - 1
        if is_main_process():
            n_v34_ae = sum(p.numel() for p in v34_e1_adapter_edit_params)
            print(
                f"[v34_E1] Group {v34_e1_adapter_edit_group_idx} "
                f"(adapter_edit, Phase 0 frozen): "
                f"{len(v34_e1_adapter_edit_params)} tensors, "
                f"{n_v34_ae/1e6:.2f}M params, "
                f"lr={v34_e1_adapter_edit_lr_phase0:.2e} (Phase 0); "
                f"will switch to {CONFIG.get('lr_adapter_edit_phase1', 5e-6):.2e} "
                f"at epoch {CONFIG.get('phase1_start_epoch', 6)}"
            )

    optimizer = optim.AdamW(param_groups_list, weight_decay=1e-2)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["phase2_start_epoch"], eta_min=5e-6
    )

    start_epoch = 1
    best_mol_acc = 0.0
    best_rc_f1 = 0.0

    best_main_acc_for_es = 0.0
    best_main_acc_epoch = 0
    epochs_without_improvement = 0
    history = []
    history_path = os.path.join(CONFIG["save_dir"], "history.csv")

    if CONFIG["resume_from_epoch"] > 0:

        best_path = f"{CONFIG['save_dir']}/model_best.pth"
        epoch_path = f"{CONFIG['save_dir']}/model_epoch_{CONFIG['resume_from_epoch']}.pth"
        if os.path.exists(best_path) and os.path.exists(epoch_path):
            best_ckpt_tmp = torch.load(best_path, map_location="cpu")
            epoch_ckpt_tmp = torch.load(epoch_path, map_location="cpu")

            if best_ckpt_tmp.get("best_mol_acc", 0) > epoch_ckpt_tmp.get("best_mol_acc", 0):
                print(f"  model_epoch_{CONFIG['resume_from_epoch']}.pth 可能已被覆盖，使用 model_best.pth")
                resume_path = best_path
            else:
                resume_path = epoch_path
            del best_ckpt_tmp, epoch_ckpt_tmp
        elif os.path.exists(best_path):
            resume_path = best_path
        else:
            resume_path = epoch_path

        if not os.path.exists(resume_path):
            phase1_path = f"{CONFIG['save_dir']}/model_phase1_final.pth"
            if (
                os.path.exists(phase1_path)
                and CONFIG["resume_from_epoch"] <= CONFIG["phase2_start_epoch"]
            ):
                print(
                    f"  epoch {CONFIG['resume_from_epoch']} checkpoint 不存在，使用 Phase 1 final checkpoint"
                )
                resume_path = phase1_path
        if os.path.exists(resume_path):
            print(f"Resuming from epoch {CONFIG['resume_from_epoch']}...")
            ckpt = torch.load(resume_path, map_location=device)
            model_dict = raw_model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in ckpt["model_state_dict"].items()
                if k in model_dict and v.shape == model_dict[k].shape
            }

            migrated = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("adapter."):
                    suffix = k[len("adapter.") :]
                    migrated[f"adapter_rc.{suffix}"] = v
                    migrated[f"adapter_edit.{suffix}"] = v
            if migrated:
                print(
                    f"  迁移 adapter → adapter_rc + adapter_edit ({len(migrated)} keys)"
                )
                ckpt["model_state_dict"].update(migrated)

                pretrained_dict = {
                    k: v
                    for k, v in ckpt["model_state_dict"].items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
            ignored = [k for k in ckpt["model_state_dict"] if k not in pretrained_dict]
            if ignored:
                print(f"  {len(ignored)} layers ignored (e.g. {ignored[:3]}...)")
            raw_model.load_state_dict(pretrained_dict, strict=False)
            if len(ignored) == 0:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass

                if "scheduler_state_dict" in ckpt:
                    try:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    except Exception:
                        pass

            if "contrastive_state_dict" in ckpt:
                try:
                    contrastive_loss_fn.load_state_dict(ckpt["contrastive_state_dict"])
                except Exception:
                    pass
            if "best_mol_acc" in ckpt:
                best_mol_acc = ckpt["best_mol_acc"]
            if "best_rc_f1" in ckpt:
                best_rc_f1 = ckpt["best_rc_f1"]
            start_epoch = CONFIG["resume_from_epoch"] + 1
            if os.path.exists(history_path):
                history = pd.read_csv(history_path).to_dict("records")

                history = [h for h in history if h["epoch"] < start_epoch]

    warmstart_path = CONFIG.get("warmstart_from", "")

    ws_phase2_initialized = False
    ws_rc_temperature = 1.0
    if not warmstart_path and start_epoch == 1:
        if is_main_process():
            print(
                "[v18] Partial scratch: warmstart_from='' — 跳过 EditGNN warmstart 链；"
                "encoder 仍从 elec_ckpt_path 加载 (./checkpoints/model_epoch_200.pth)"
            )
    if warmstart_path and start_epoch == 1 and os.path.exists(warmstart_path):
        if is_main_process():
            print(f"Warmstart: loading model weights from {warmstart_path}...")
        ws_ckpt = torch.load(warmstart_path, map_location=device)
        ws_sd = ws_ckpt["model_state_dict"]

        v15_remapped = 0
        remapped_sd = {}
        for k, v in ws_sd.items():
            if "frag_full_attn." in k and "frag_full_attn.layers." not in k:
                new_k = k.replace("frag_full_attn.", "frag_full_attn.layers.0.", 1)
                remapped_sd[new_k] = v
                v15_remapped += 1
            else:
                remapped_sd[k] = v
        ws_sd = remapped_sd
        if is_main_process() and v15_remapped > 0:
            print(f"  v15 key-rename: frag_full_attn.* → frag_full_attn.layers.0.* ({v15_remapped} tensors)")

        v17_1_ga_remapped = 0
        if CONFIG.get("n_global_anchor_layers", 1) > 0:
            remapped_sd = {}
            for k, v in ws_sd.items():
                if ".global_anchor." in k                   and ".global_anchor.0." not in k                   and ".global_anchor.1." not in k                   and ".global_anchor.2." not in k:
                    new_k = k.replace(".global_anchor.", ".global_anchor.0.", 1)
                    remapped_sd[new_k] = v
                    v17_1_ga_remapped += 1
                else:
                    remapped_sd[k] = v
            ws_sd = remapped_sd
            if is_main_process() and v17_1_ga_remapped > 0:
                print(
                    f"  v17.1 GlobalAnchor remap: {v17_1_ga_remapped} tensors "
                    f"rc_head.global_anchor.* → rc_head.global_anchor.0.* "
                    f"(ModuleList layer 0 继承 v16.1，layer 1+ fresh init)"
                )

        model_dict = raw_model.state_dict()
        ws_dict = {}
        shape_mismatch = []
        unexpected = []
        zero_padded = []

        v19_pad_key_patterns = (
            "rc_head.fc1.weight",
            "rc_head.skip_proj.weight",
            "frag_descriptor.proj.0.weight",
            "frag_descriptor.proj.0.bias",
            "frag_descriptor.proj.1.weight",
        )
        pad_init_scale = float(CONFIG.get("warmstart_pad_init_scale", 0.1))
        for k, v in ws_sd.items():
            if k not in model_dict:
                unexpected.append(k)
            elif v.shape != model_dict[k].shape:
                target_shape = model_dict[k].shape

                if len(v.shape) == len(target_shape) and all(
                    ts >= vs for ts, vs in zip(target_shape, v.shape)
                ):
                    padded = model_dict[k].clone()

                    if any(pat in k for pat in v19_pad_key_patterns) and v.dim() == 2:

                        new_region_slices = tuple(
                            slice(vs, ts) if ts > vs else slice(0, 0)
                            for vs, ts in zip(v.shape, target_shape)
                        )
                        if any(s.stop > s.start for s in new_region_slices):
                            padded[new_region_slices] = padded[new_region_slices] * pad_init_scale
                    slices = tuple(slice(0, s) for s in v.shape)
                    padded[slices] = v
                    ws_dict[k] = padded
                    zero_padded.append((k, tuple(v.shape), tuple(target_shape)))
                else:
                    shape_mismatch.append((k, tuple(v.shape), tuple(target_shape)))
            else:
                ws_dict[k] = v
        missing = [k for k in model_dict.keys() if k not in ws_sd]
        raw_model.load_state_dict(ws_dict, strict=False)

        if (CONFIG.get("use_detr_edit_head", False)
                and CONFIG.get("detr_edit_init_from_v29_weight", True)
                and getattr(raw_model, "detr_edit_head", None) is not None):
            mlp_sd_for_detr = {
                k.replace("edit_classifier.", "", 1): v
                for k, v in ws_sd.items()
                if k.startswith("edit_classifier.mlp.")
            }
            if mlp_sd_for_detr:
                try:
                    n_copied = raw_model.detr_edit_head.init_edit_classifier_from_v29(
                        mlp_sd_for_detr
                    )
                    if is_main_process():
                        print(
                            f"  [v34_E1] init_from_v29: copied {n_copied} tensors "
                            f"v29.2 EditClassifier.mlp.* → detr_edit_head.edit_classifier"
                        )
                except Exception as e:
                    if is_main_process():
                        print(f"  [v34_E1] init_from_v29 FAILED: {type(e).__name__}: {e}")
            else:
                if is_main_process():
                    print(
                        "  [v34_E1] init_from_v29 SKIPPED: no edit_classifier.mlp.* "
                        "keys in warmstart ckpt"
                    )

        identity_init_count = 0
        identity_skip_count = 0
        for layer_idx in range(1, CONFIG.get("frag_full_attn_layers", 2)):
            try:
                layer = raw_model.rc_head.frag_full_attn.layers[layer_idx]

                out_proj_norm = layer.mha.out_proj.weight.norm().item()
                ffn_last_norm = layer.ffn[-1].weight.norm().item()
                trained_threshold = 0.01
                if out_proj_norm > trained_threshold and ffn_last_norm > trained_threshold:
                    identity_skip_count += 1
                    if is_main_process():
                        print(f"  v15.1: layer {layer_idx} 已有训练权重 (out_proj norm={out_proj_norm:.3f}, ffn_last norm={ffn_last_norm:.3f}), 跳过 identity-init")
                    continue
                with torch.no_grad():
                    layer.mha.out_proj.weight.zero_()
                    if layer.mha.out_proj.bias is not None:
                        layer.mha.out_proj.bias.zero_()

                    last_lin = layer.ffn[-1]
                    last_lin.weight.zero_()
                    last_lin.bias.zero_()
                identity_init_count += 1
            except (AttributeError, IndexError) as e:
                if is_main_process():
                    print(f"  v15 identity-init 跳过 layer {layer_idx}: {type(e).__name__}: {e}")
        if is_main_process() and identity_init_count > 0:
            print(f"  v15 identity-init: FragmentFullAttention 新增 {identity_init_count} 层 out_proj/ffn_last 置零")

        loaded = len(ws_dict)
        total = len(model_dict)
        if is_main_process():
            print(f"  Warmstart loaded {loaded}/{total} tensors")
            if zero_padded:
                print(f"  Zero-padded (v14 扩展保留旧权重): {len(zero_padded)}")
                for k, sv, mv in zero_padded:
                    print(f"    + {k}: {sv} → {mv}")
            if shape_mismatch:
                print(f"  Shape-mismatch (fresh init): {len(shape_mismatch)}")
                for k, sv, mv in shape_mismatch[:8]:
                    print(f"    - {k}: ckpt={sv} model={mv}")
                if len(shape_mismatch) > 8:
                    print(f"    ... and {len(shape_mismatch) - 8} more")
            if missing:
                print(f"  Missing in ckpt (fresh init): {len(missing)}")

                new_prefixes = ("pair_frag_attn", "fc1", "fc2", "fc3",
                                "skip_proj", "frag_full_attn",
                                "global_anchor", "pair_sym_head",
                                "pair_hard_head",
                                "pair_cons_gate", "fukui_proj",
                                "charge_proj", "fmo_proj")
                highlights = [k for k in missing if any(k.startswith(p) or p in k for p in new_prefixes)]
                for k in highlights[:12]:
                    print(f"    + {k}")
                if len(highlights) < len(missing):
                    print(f"    ... ({len(missing) - len(highlights)} other missing keys silently accepted)")
            if unexpected:
                print(f"  Unexpected in ckpt (skipped): {len(unexpected)}")
                for k in unexpected[:5]:
                    print(f"    - {k}")

        if getattr(raw_model, "multi_hop_rc_gate", None) is not None:
            with torch.no_grad():
                raw_model.multi_hop_rc_gate.fill_(0.05)
                if getattr(raw_model, "multi_hop_rc_proj", None) is not None:
                    nn.init.kaiming_uniform_(raw_model.multi_hop_rc_proj.weight, a=math.sqrt(5))
                    if raw_model.multi_hop_rc_proj.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_model.multi_hop_rc_proj.weight)
                        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(raw_model.multi_hop_rc_proj.bias, -bound, bound)
            if is_main_process():
                gate_val = raw_model.multi_hop_rc_gate.detach().cpu().tolist()
                print(f"  [v27.1 hotfix] multi_hop_rc reset after warmstart: gate={gate_val}, proj re-initialized")

        ws_phase2_initialized = bool(ws_ckpt.get("phase2_initialized", False))
        ws_rc_temperature = float(ws_ckpt.get("rc_temperature", 1.0))
        if is_main_process() and ws_phase2_initialized:
            print(
                f"  v20.5: warmstart ckpt phase2_initialized=True, rc_temperature={ws_rc_temperature:.4f} "
                f"→ 跳过阈值搜索/温度校准, 保留 CONFIG ss_rc_threshold={CONFIG['ss_rc_threshold']}"
            )
        del ws_ckpt, ws_sd, ws_dict

    if is_main_process():
        print(f"Start EditGNN Training: {len(train_ds)} train, {len(val_ds)} val")
        print(f"Phase 1 (RC only): epochs 1-{CONFIG['phase2_start_epoch']-1}")
        print(f"Phase 2 (RC + Edit): epochs {CONFIG['phase2_start_epoch']}+")

    phase2_initialized = start_epoch > CONFIG["phase2_start_epoch"]
    nan_epoch_count = 0
    last_good_epoch = None
    edit_collapse_count = 0

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):

        if CONFIG.get("use_equivalence_class_aug", False):
            _aug_warmup = int(CONFIG.get("equiv_aug_warmup_epochs", 0))
            _aug_prob_target = float(CONFIG.get("equiv_aug_prob", 0.0))
            CONFIG["equiv_aug_prob_active"] = 0.0 if epoch <= _aug_warmup else _aug_prob_target
            if is_main_process():
                _state = "WARMUP(0.0)" if epoch <= _aug_warmup else f"ACTIVE({_aug_prob_target:.2f})"
                print(f"  [v29.2 G] equiv_aug_prob_active = {CONFIG['equiv_aug_prob_active']:.2f} {_state}")

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        if CONFIG.get("use_pair_cons_gate", False) and getattr(raw_model, "pair_cons_gate", None) is not None:
            raw_model.pair_cons_gate.set_alpha_from_warmup(
                epoch,
                int(CONFIG.get("pair_cons_warmup_start", 0)),
                int(CONFIG.get("pair_cons_warmup_end", 3)),
            )
            if is_main_process():
                alpha_val = float(raw_model.pair_cons_gate.alpha.detach().item())
                print(f"  [v19] pair_cons_gate.alpha = {alpha_val:.4f} (ep{epoch})")

        if CONFIG.get("use_multi_hop_rc", False) and hasattr(raw_model, "_multi_hop_rc_warmup_active"):
            warmup_end = int(CONFIG.get("multi_hop_rc_warmup_epochs", 2))
            new_state = (epoch <= warmup_end)
            if raw_model._multi_hop_rc_warmup_active != new_state:
                raw_model._multi_hop_rc_warmup_active = new_state
                if is_main_process():
                    state_str = "WARMUP (single hop = v26)" if new_state else "ACTIVE (multi-hop iteration)"
                    print(f"  [v27 A1] multi_hop_rc transition @ ep{epoch}: {state_str}")

            if raw_model.multi_hop_rc_gate is not None:
                raw_model.multi_hop_rc_gate.requires_grad_(not new_state)

        if CONFIG.get("cross_frag_layers", 2) > 2 and getattr(raw_model, "cross_frag_attn", None) is not None:
            cf_freeze_end = int(CONFIG.get("cross_frag_attn_freeze_epochs", 0))
            cf_frozen = (epoch <= cf_freeze_end)

            for layer_idx in range(min(2, CONFIG.get("cross_frag_layers", 2))):
                attn = raw_model.cross_frag_attn.attn_layers[layer_idx]
                ffn = raw_model.cross_frag_attn.ffn_layers[layer_idx]
                norm1 = raw_model.cross_frag_attn.norm1_layers[layer_idx]
                norm2 = raw_model.cross_frag_attn.norm2_layers[layer_idx]
                for mod in (attn, ffn, norm1, norm2):
                    for p in mod.parameters():
                        p.requires_grad_(not cf_frozen)
            if is_main_process() and epoch == cf_freeze_end + 1:
                print(f"  [v27 B1] cross_frag_attn old layers UNFROZEN @ ep{epoch}")

        current_phase = 1 if epoch < CONFIG["phase2_start_epoch"] else 2

        if CONFIG.get("current_phase_force", 0) >= 2:
            current_phase = max(current_phase, 2)

        _v34_p1_start = CONFIG.get("phase1_start_epoch", 6)
        if epoch == _v34_p1_start:
            if v34_e1_detr_edit_group_idx is not None:
                _new_detr_lr = float(CONFIG.get("lr_detr_edit_phase1", 1e-4))
                optimizer.param_groups[v34_e1_detr_edit_group_idx]["lr"] = _new_detr_lr
                if "initial_lr" in optimizer.param_groups[v34_e1_detr_edit_group_idx]:
                    optimizer.param_groups[v34_e1_detr_edit_group_idx]["initial_lr"] = _new_detr_lr
                if is_main_process():
                    print(
                        f"  [v34_E1] Phase 0 → Phase 1 @ ep{epoch}: "
                        f"DETREditHead LR → {_new_detr_lr:.2e} (was 5e-4)"
                    )
            if v34_e1_adapter_edit_group_idx is not None:
                _new_ae_lr = float(CONFIG.get("lr_adapter_edit_phase1", 5e-6))
                optimizer.param_groups[v34_e1_adapter_edit_group_idx]["lr"] = _new_ae_lr
                if "initial_lr" in optimizer.param_groups[v34_e1_adapter_edit_group_idx]:
                    optimizer.param_groups[v34_e1_adapter_edit_group_idx]["initial_lr"] = _new_ae_lr
                if is_main_process():
                    print(
                        f"  [v34_E1] Phase 0 → Phase 1 @ ep{epoch}: "
                        f"adapter_edit UNFROZEN, LR → {_new_ae_lr:.2e} (was 0.0)"
                    )

        if (
            CONFIG.get("use_large_cand_dropout", False)
            and current_phase == 2
        ):
            _phase2_ep = epoch - CONFIG["phase2_start_epoch"]
            _lcd_warmup = CONFIG.get("large_cand_dropout_warmup_eps", 5)
            _lcd_max = CONFIG.get("large_cand_dropout_max", 0.3)
            if _phase2_ep < _lcd_warmup:
                _lcd_p = _lcd_max * (1.0 - _phase2_ep / max(_lcd_warmup, 1))
            else:
                _lcd_p = 0.0
            raw_model.large_cand_dropout_p = float(_lcd_p)
            if is_main_process() and _lcd_p > 0:
                print(
                    f"  [v20.2] large_cand_dropout_p = {_lcd_p:.4f} "
                    f"(phase2_ep={_phase2_ep}, warmup={_lcd_warmup})"
                )
        else:
            raw_model.large_cand_dropout_p = 0.0

        if current_phase == 2 and not phase2_initialized:

            phase1_save = f"{CONFIG['save_dir']}/model_phase1_final.pth"
            if is_main_process():
                torch.save(
                    {
                        "epoch": epoch - 1,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "contrastive_state_dict": contrastive_loss_fn.state_dict(),
                        "best_mol_acc": best_mol_acc,
                        "phase2_initialized": False,
                    },
                    phase1_save,
                )
                print(f"\n>>> Phase 1 最终 checkpoint 已保存: {phase1_save}")

            if ws_phase2_initialized:
                if is_main_process():
                    print(
                        f">>> v20.5 跳过阈值搜索 (warmstart Phase 2 ckpt): "
                        f"使用 CONFIG ss_rc_threshold={CONFIG['ss_rc_threshold']}, "
                        f"ws_rc_temperature={ws_rc_temperature}"
                    )
                optimal_thr = float(CONFIG["ss_rc_threshold"])
                rc_temperature = ws_rc_temperature
            elif is_main_process():
                print(">>> Phase 1 结束，搜索最优 RC 阈值...")
                thr_loader = DataLoader(
                    val_ds,
                    batch_size=CONFIG["loader_batch_size"],
                    shuffle=False,
                    num_workers=CONFIG["num_workers"],
                    collate_fn=editgnn_collate_eval,
                    pin_memory=True,
                )
                optimal_thr = find_optimal_rc_threshold(raw_model, thr_loader, device)

                print(">>> 温度缩放校准...")
                rc_temperature = calibrate_temperature(raw_model, thr_loader, device)
                del thr_loader
            else:
                optimal_thr = 0.5
                rc_temperature = 1.0
            if is_ddp:
                thr_tensor = torch.tensor(optimal_thr, device=device)
                dist.broadcast(thr_tensor, src=0)
                optimal_thr = thr_tensor.item()
                temp_tensor = torch.tensor(rc_temperature, device=device)
                dist.broadcast(temp_tensor, src=0)
                rc_temperature = temp_tensor.item()
            CONFIG["ss_rc_threshold"] = optimal_thr
            CONFIG["rc_temperature"] = rc_temperature

            current_rc_lr = optimizer.param_groups[1]["lr"]
            new_rc_lr = current_rc_lr * CONFIG["phase2_lr_factor"]
            if is_main_process():
                print(
                    f"\n>>> Phase 1 -> 2 过渡: "
                    f"RC 组 LR {current_rc_lr:.1e} -> {new_rc_lr:.1e}, "
                    f"Edit 组 LR -> {CONFIG['lr_edit']:.1e}, "
                    f"RC 冻结 {CONFIG['phase2_rc_freeze_epochs']} epochs, "
                    f"edit loss warmup {CONFIG['phase2_warmup_epochs']} epochs, "
                    f"RC pos_weight {CONFIG['rc_pos_weight']} -> {CONFIG['phase2_rc_pos_weight']}, "
                    f"SS 阈值={optimal_thr:.2f}, SS prob 0→{CONFIG['ss_end_prob']}"
                )
            optimizer.param_groups[1]["lr"] = new_rc_lr
            optimizer.param_groups[2]["lr"] = CONFIG["lr_edit"]

            for p in optimizer.param_groups[2]["params"]:
                state = optimizer.state[p]
                if state:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

            optimizer.param_groups[0]["initial_lr"] = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[1]["initial_lr"] = new_rc_lr
            optimizer.param_groups[2]["initial_lr"] = CONFIG["lr_edit"]

            if len(optimizer.param_groups) > 3:
                encoder_top_lr = CONFIG.get("lr_encoder_top", 3e-5) * CONFIG["phase2_lr_factor"]
                optimizer.param_groups[3]["lr"] = encoder_top_lr
                optimizer.param_groups[3]["initial_lr"] = encoder_top_lr

            if len(optimizer.param_groups) > 4:
                cross_frag_lr = CONFIG.get("lr_cross_frag", 3e-4) * CONFIG["phase2_lr_factor"]
                optimizer.param_groups[4]["lr"] = cross_frag_lr
                optimizer.param_groups[4]["initial_lr"] = cross_frag_lr

            if len(optimizer.param_groups) > 5:
                v16_lr_phase2 = CONFIG["lr_rc"] * CONFIG.get("new_module_lr_mult", 1.5) * CONFIG["phase2_lr_factor"]
                optimizer.param_groups[5]["lr"] = v16_lr_phase2
                optimizer.param_groups[5]["initial_lr"] = v16_lr_phase2

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CONFIG.get("cosine_phase2_T_max", 30),
                eta_min=5e-6,
            )

            phase2_initialized = True

        if current_phase == 2:
            epochs_in_phase2 = epoch - CONFIG["phase2_start_epoch"]

            edit_ramp_ceiling = float(CONFIG.get("edit_ramp_ceiling", 1.0))
            edit_ramp = min(
                edit_ramp_ceiling, (epochs_in_phase2 + 1) / CONFIG["phase2_warmup_epochs"]
            )

            ss_prob = min(
                CONFIG["ss_end_prob"],
                CONFIG["ss_start_prob"]
                + (CONFIG["ss_end_prob"] - CONFIG["ss_start_prob"])
                * epochs_in_phase2
                / max(CONFIG["ss_warmup_epochs"], 1),
            )

            rc_frozen = epochs_in_phase2 < CONFIG["phase2_rc_freeze_epochs"]

            refine_start = CONFIG.get("rc_refine_start_epoch", 5)
            refine_warmup = CONFIG.get("rc_refine_warmup_epochs", 10)
            if CONFIG.get("rc_refine_enabled", True) and epochs_in_phase2 >= refine_start:
                refine_epochs_active = epochs_in_phase2 - refine_start
                use_rc_refine = min(1.0, refine_epochs_active / max(refine_warmup, 1))
            else:
                use_rc_refine = 0.0
        else:
            edit_ramp = 0.0
            ss_prob = 0.0
            rc_frozen = False
            use_rc_refine = 0.0

        if current_phase == 1:
            for p in optimizer.param_groups[2]["params"]:
                p.requires_grad = False
        else:
            for p in optimizer.param_groups[2]["params"]:
                p.requires_grad = True

        rc_modules_params = (
            list(raw_model.adapter_rc.parameters())
            + list(raw_model.rc_head.parameters())
        )
        if getattr(raw_model, "pfrh", None) is not None:
            rc_modules_params += list(raw_model.pfrh.parameters())
        if rc_frozen:
            for p in rc_modules_params:
                p.requires_grad = False
        else:
            for p in rc_modules_params:
                p.requires_grad = True

        phase_str = f"Phase {current_phase}"
        if current_phase == 2:
            phase_str += f" ramp={edit_ramp:.2f} ss={ss_prob:.2f}"
            if rc_frozen:
                phase_str += " RC_frozen"
            if use_rc_refine > 0:
                phase_str += f" RC_refine={use_rc_refine:.2f}"

        model.train()
        train_tracker = {
            "loss": 0,
            "l_rc": 0,
            "l_edit": 0,
            "l_contrastive": 0,
            "l_refined_rc": 0,
            "l_pair_sym": 0,
            "l_bond_rc": 0,
            "l_pair_contrastive": 0,
            "l_rc_count": 0,
            "l_pair_head": 0,
            "l_pair_teach": 0,
            "l_conf_pen": 0,
            "l_all_atom_hit": 0,
            "l_pair_hard": 0,
            "l_valence_rc": 0,
            "l_pair_exist": 0,
            "l_fukui_rank": 0,
            "l_mol_aux": 0,
            "l_cc_pair": 0,
            "l_set_recall": 0,
            "l_set_precision": 0,
            "l_pointer_rc": 0,
            "l_valence_dh": 0,
            "l_margin_penalty": 0,
            "l_mutual_exclusion": 0,
            "l_edit_cond_rc": 0,
            "l_l1_sparsity": 0,
            "l_k_aware_sparsity": 0,
            "l_k_pred": 0,
            "k_pred_acc": 0,
            "pair_cons_alpha": 0,

            "l_rc_set_pred": 0,
            "l_rc_set_atom_ce": 0,
            "l_rc_set_presence": 0,
            "rc_set_matched": 0,
            "rc_set_no_object": 0,
            "rc_recall": 0,
            "rc_precision": 0,
            "rc_f1": 0,
            "rc_both_rate": 0,
            "all_atom_hit_rate": 0,
            "rc_avg_prob_pos": 0,
            "rc_avg_prob_neg": 0,
            "rc_pred_count": 0,
            "rc_gt_count": 0,
            "edit_acc": 0,
            "edit_recall": 0,
            "edit_precision": 0,
            "edit_conf_correct": 0,
            "edit_conf_wrong": 0,
            "edit_nonzero_pred_count": 0,
            "edit_nonzero_gt_count": 0,
            "cand_nonzero_ratio": 0,
            "mol_acc": 0,
            "mol_fail_rc": 0,
            "mol_fail_edit": 0,
            "mol_fail_coverage": 0,
            "cand_gt_coverage": 0,
            "avg_cand_edges": 0,

            "K0_mol_acc": 0, "K0_count": 0,
            "K2_mol_acc": 0, "K2_count": 0,
            "K3_mol_acc": 0, "K3_count": 0,
            "K4_mol_acc": 0, "K4_count": 0,
            "K5p_mol_acc": 0, "K5p_count": 0,
            "main_mol_acc": 0,

            "main_mol_acc_revised": 0,
            "K_pos_main_mol_acc": 0, "K_pos_count": 0,
            "K4plus_main_mol_acc": 0, "K4plus_count": 0,
            "bucket_avg_acc": 0,
            "fail_rc_total": 0,
            "K0_main_revised": 0,
            "K2_main_revised": 0,
            "K3_main_revised": 0,
            "K4_main_revised": 0,
            "K5p_main_revised": 0,

            "main_mol_acc_product": 0,
            "K_pos_product_mol_acc": 0,
            "K0_product": 0,
            "K2_product": 0,
            "K3_product": 0,
            "K4_product": 0,
            "K5p_product": 0,
        }

        last_edit_per_class_acc = ""
        steps = 0
        nan_count = 0
        nan_batch_total = 0
        spike_batch_total = 0
        grad_norm_accum = [0.0, 0.0, 0.0]
        grad_norm_steps = 0

        epoch_rc_sym_stats = torch.zeros(8, dtype=torch.long)
        loss_ema = None
        loss_ema_warmup = 0
        optimizer.zero_grad()
        pbar = tqdm(
            train_loader,
            desc=f"Ep {epoch} [{phase_str}]",
            leave=False,
            disable=not is_main_process(),
        )

        for i, batch in enumerate(pbar):
            if batch is None:
                continue

            _bs = getattr(batch, "rc_sym_stats", None)
            if _bs is not None:
                epoch_rc_sym_stats += _bs.long()
            batch = batch.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    batch,
                    training=True,
                    phase=current_phase,
                    ss_prob=ss_prob,
                    rc_refine=use_rc_refine,
                )

            loss, l_dict = compute_loss(
                outputs,
                batch,
                CONFIG,
                current_phase,
                edit_ramp,
                model=raw_model,
                epoch=epoch,
                contrastive_loss_fn=contrastive_loss_fn,
            )

            if current_phase >= 2:
                _edit_logits = outputs.get("edit_logits")
                if _edit_logits is not None and _edit_logits.numel() > 0 and not torch.isfinite(_edit_logits).all():
                    nan_batch_total += 1
                    if is_main_process() and nan_batch_total <= 3:
                        print(f"\n  !! edit_logits NaN/Inf @ ep{epoch} batch {i}, skip (shape={tuple(_edit_logits.shape)})")
                    continue

            current_loss = loss.item()
            if not math.isfinite(current_loss):
                nan_count += 1
                nan_batch_total += 1

                if nan_count >= CONFIG["nan_patience"]:
                    if is_main_process():
                        print(
                            f"\n  !! 连续 {nan_count} 次 NaN/Inf loss，终止当前 epoch"
                        )
                    break
                continue
            nan_count = 0

            if loss_ema is None:
                loss_ema = current_loss
                loss_ema_warmup = 1
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * current_loss
                loss_ema_warmup += 1
                if loss_ema_warmup >= 10 and current_loss > loss_ema * CONFIG["loss_spike_ratio"]:
                    spike_batch_total += 1
                    if is_main_process():
                        print(
                            f"\n  !! Loss 突增: {current_loss:.3f} > {loss_ema:.3f} * {CONFIG['loss_spike_ratio']}, 跳过此 batch"
                        )

                    continue

            (loss / CONFIG["accum_steps"]).backward()

            steps += 1

            with torch.no_grad():
                batch_metrics = compute_metrics(outputs, batch, CONFIG)

            train_tracker["loss"] += current_loss
            for k, v in l_dict.items():
                train_tracker[k] += v
            for k, v in batch_metrics.items():
                if k == "edit_per_class_acc":
                    last_edit_per_class_acc = v
                else:
                    train_tracker[k] += v

            if steps > 0 and steps % CONFIG["accum_steps"] == 0:

                _clip_norms = [1.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 3.0] + [5.0] * max(0, len(optimizer.param_groups) - 8)
                grad_norms = []
                has_nan_grad = False
                for pg, mn in zip(optimizer.param_groups, _clip_norms):
                    gn = torch.nn.utils.clip_grad_norm_(pg["params"], mn)
                    gn_val = gn.item()
                    grad_norms.append(gn_val)
                    if not math.isfinite(gn_val):
                        has_nan_grad = True
                if has_nan_grad:

                    if i < 3 and is_main_process():
                        nan_names = []
                        for name, p in raw_model.named_parameters():
                            if p.grad is not None and not torch.isfinite(p.grad).all():
                                nan_names.append(name)
                                if len(nan_names) >= 3:
                                    break
                        print(f"  NaN_GRAD (first 3 of all nan params): {nan_names}")

                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                    for gi in range(min(3, len(grad_norms))):
                        grad_norm_accum[gi] += grad_norms[gi]
                    grad_norm_steps += 1
                pbar.set_postfix(
                    {
                        "L": f"{current_loss:.3f}",
                        "RC_R": f"{batch_metrics['rc_recall']:.2f}",
                        "E_R": f"{batch_metrics['edit_recall']:.2f}",
                        "MA": f"{batch_metrics['mol_acc']:.2f}",
                        "gn": f"{grad_norms[1]:.2e}/{grad_norms[2]:.2e}",
                    }
                )

        if steps > 0 and steps % CONFIG["accum_steps"] != 0:
            _clip_norms = [1.0, 5.0, 5.0, 1.0, 5.0, 5.0, 5.0, 3.0] + [5.0] * max(0, len(optimizer.param_groups) - 8)
            grad_norms = []
            has_nan_grad = False
            for pg, mn in zip(optimizer.param_groups, _clip_norms):
                gn = torch.nn.utils.clip_grad_norm_(pg["params"], mn)
                gn_val = gn.item()
                grad_norms.append(gn_val)
                if not math.isfinite(gn_val):
                    has_nan_grad = True
            if not has_nan_grad:
                optimizer.step()
            optimizer.zero_grad()

        if steps == 0:

            nan_epoch_count += 1
            if is_main_process():
                print(
                    f"  !! Epoch {epoch} 全部 NaN (连续 {nan_epoch_count} 个 NaN epoch)"
                )
            if (
                nan_epoch_count >= CONFIG["nan_epoch_patience"]
                and last_good_epoch is not None
            ):
                rollback_path = (
                    f"{CONFIG['save_dir']}/model_epoch_{last_good_epoch}.pth"
                )
                if os.path.exists(rollback_path):
                    if is_main_process():
                        print(
                            f"  >> 自动回滚到 epoch {last_good_epoch} checkpoint，降低 edit LR"
                        )
                    ckpt = torch.load(rollback_path, map_location=device)
                    raw_model.load_state_dict(ckpt["model_state_dict"])
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    if "scheduler_state_dict" in ckpt:
                        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                    optimizer.param_groups[2]["lr"] *= 0.5
                    if is_main_process():
                        print(
                            f"     edit LR 降为 {optimizer.param_groups[2]['lr']:.1e}"
                        )
                    nan_epoch_count = 0
                else:
                    if is_main_process():
                        print(f"  !! 回滚 checkpoint 不存在: {rollback_path}，继续训练")
            continue

        nan_epoch_count = 0
        last_good_epoch = epoch
        train_avg = {k: v / steps for k, v in train_tracker.items()}
        train_avg["edit_per_class_acc"] = last_edit_per_class_acc

        model.eval()
        val_tracker = {k: 0 for k in train_tracker}
        val_last_edit_per_class_acc = ""
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                batch = batch.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(
                        batch,
                        training=False,
                        phase=current_phase,
                        rc_refine=use_rc_refine,
                    )
                loss, l_dict = compute_loss(
                    outputs,
                    batch,
                    CONFIG,
                    current_phase,
                    edit_ramp,
                    model=raw_model,
                    epoch=epoch,
                    contrastive_loss_fn=contrastive_loss_fn,
                )

                if not math.isfinite(loss.item()):
                    continue
                val_steps += 1
                batch_metrics = compute_metrics(outputs, batch, CONFIG)
                val_tracker["loss"] += loss.item()
                for k, v in l_dict.items():
                    val_tracker[k] += v
                for k, v in batch_metrics.items():
                    if k == "edit_per_class_acc":
                        val_last_edit_per_class_acc = v
                    else:
                        val_tracker[k] += v

        if val_steps > 0:
            val_avg = {k: v / val_steps for k, v in val_tracker.items()}
        else:
            val_avg = {k: 0 for k in val_tracker}
        val_avg["edit_per_class_acc"] = val_last_edit_per_class_acc

        if is_ddp:

            keys_to_reduce = [k for k in val_avg if k != "edit_per_class_acc"]
            vals_tensor = torch.tensor(
                [val_avg[k] for k in keys_to_reduce], device=device
            )
            dist.all_reduce(vals_tensor, op=dist.ReduceOp.AVG)
            for k, v in zip(keys_to_reduce, vals_tensor.tolist()):
                val_avg[k] = v

        if is_main_process():
            lrs = [pg["lr"] for pg in optimizer.param_groups]

            kpp = val_avg.get('K_pos_product_mol_acc', 0.0)
            kpm = val_avg.get('K_pos_main_mol_acc', 0.0)
            best_str = f" | best={best_mol_acc:.4f}" if best_mol_acc > 0 else ""
            print(
                f"Ep {epoch} [{phase_str}] Loss: T={train_avg['loss']:.3f} V={val_avg['loss']:.3f} | "
                f"K_pos: product={kpp:.4f} main={kpm:.4f}{best_str}"
            )

            print(
                f"  RC: R={val_avg['rc_recall']:.1%} P={val_avg['rc_precision']:.1%} F1={val_avg['rc_f1']:.1%} | "
                f"Edit: R={val_avg['edit_recall']:.1%} P={val_avg['edit_precision']:.1%} | "
                f"MolAcc(strict)={val_avg['mol_acc']:.1%} Main={val_avg.get('main_mol_acc',0):.1%}"
            )
            if current_phase >= 2:

                kp0 = val_avg.get('K0_product', val_avg.get('K0_main_revised', val_avg.get('K0_mol_acc', 0)))
                kp2 = val_avg.get('K2_product', val_avg.get('K2_main_revised', val_avg.get('K2_mol_acc', 0)))
                kp3 = val_avg.get('K3_product', val_avg.get('K3_main_revised', val_avg.get('K3_mol_acc', 0)))
                kp4 = val_avg.get('K4_product', val_avg.get('K4_main_revised', val_avg.get('K4_mol_acc', 0)))
                kp5 = val_avg.get('K5p_product', val_avg.get('K5p_main_revised', val_avg.get('K5p_mol_acc', 0)))
                print(
                    f"  Per-K (product): K0={kp0:.1%} K2={kp2:.1%} K3={kp3:.1%} K4={kp4:.1%} K5+={kp5:.1%}"
                )

            v27_status_parts = []
            if getattr(raw_model, "multi_hop_rc_gate", None) is not None:
                g = raw_model.multi_hop_rc_gate.detach().cpu().tolist()
                wu = "WARMUP" if raw_model._multi_hop_rc_warmup_active else "ACTIVE"
                g_str = "/".join(f"{x:+.3f}" for x in g)
                v27_status_parts.append(f"A1 multi_hop_rc[{wu}]={g_str}")
            if CONFIG.get("use_k_aware_sparsity_loss", False):
                k_aware_val = val_avg.get("l_k_aware_sparsity", 0.0)
                wu_end = int(CONFIG.get("k_aware_warmup_epochs", 3))
                wu_state = "WARMUP" if epoch < wu_end else "ACTIVE"
                v27_status_parts.append(f"A2 K_aware[{wu_state}]={k_aware_val:.4f}")
            if CONFIG.get("cross_frag_layers", 2) > 2:
                cf_freeze_end = int(CONFIG.get("cross_frag_attn_freeze_epochs", 0))
                cf_state = "FROZEN" if epoch <= cf_freeze_end else "TRAINING"
                v27_status_parts.append(f"B1 cross_frag[{cf_state}]({CONFIG['cross_frag_layers']}L)")

            if CONFIG.get("use_k_pred_head", False):
                kp_wu_end = int(CONFIG.get("k_pred_warmup_epochs", 3))
                kp_state = "WARMUP" if epoch < kp_wu_end else "ACTIVE"
                kp_loss = val_avg.get("l_k_pred", 0.0)
                kp_acc = val_avg.get("k_pred_acc", 0.0)
                v27_status_parts.append(
                    f"v28 K_pred[{kp_state}] loss={kp_loss:.3f} acc={kp_acc:.1%}"
                )
            if v27_status_parts:
                print(f"  v27/28: " + " | ".join(v27_status_parts))

            if current_phase >= 2:
                print(
                    f"  Diag: GT_cov={val_avg['cand_gt_coverage']:.1%} "
                    f"RC_prob[+={val_avg['rc_avg_prob_pos']:.3f} -={val_avg['rc_avg_prob_neg']:.3f}] "
                    f"EditConf[ok={val_avg['edit_conf_correct']:.3f} bad={val_avg['edit_conf_wrong']:.3f}] "
                    f"NaN/Spike={nan_batch_total}/{spike_batch_total}"
                )

            lr_str = f"bert={lrs[0]:.1e} rc={lrs[1]:.1e} edit={lrs[2]:.1e}"
            if len(lrs) > 4:
                lr_str += f" cf={lrs[4]:.1e}"
            print(f"  LR: {lr_str}")

            if CONFIG.get("use_rc_sym_stats_logging", False):
                _s = epoch_rc_sym_stats.tolist()
                _n_mol = max(_s[0], 1)
                _n_pos = max(_s[5], 1)
                _n_swap = max(_s[6], 1)
                _skip_K_rate = _s[1] / _n_mol
                _no_gt_rate  = _s[2] / _n_mol
                _bad_data_rate = (_s[3] + _s[4]) / _n_mol
                _accept_rate = _s[7] / _n_swap
                _expansion_ratio = _s[7] / _n_pos
                print(
                    f"  E.B[rank0]: mol={_s[0]} skip_K={_skip_K_rate:.1%} "
                    f"no_gt={_no_gt_rate:.1%} bad_data={_bad_data_rate:.1%} | "
                    f"pos_atoms={_s[5]} swap_attempts={_s[6]} accept={_s[7]} "
                    f"accept_rate={_accept_rate:.1%} expansion_ratio={_expansion_ratio:.3f}"
                )

        if current_phase == 2:

            if val_avg["rc_recall"] < 0.5 and is_main_process():
                print(
                    f"  !! RC recall 崩溃至 {val_avg['rc_recall']:.1%}，训练不稳定，建议检查"
                )

            if val_avg["edit_acc"] < CONFIG["edit_collapse_threshold"]:
                old_lr = optimizer.param_groups[2]["lr"]
                new_lr = old_lr * 0.3
                optimizer.param_groups[2]["lr"] = new_lr
                if is_main_process():
                    print(
                        f"  !! Edit mode collapse 检测: val_edit_acc={val_avg['edit_acc']:.1%} "
                        f"< {CONFIG['edit_collapse_threshold']:.0%}"
                    )
                    print(f"     edit LR {old_lr:.1e} -> {new_lr:.1e}")
                edit_collapse_count += 1
                if edit_collapse_count >= 2:
                    phase1_ckpt = f"{CONFIG['save_dir']}/model_phase1_final.pth"
                    if os.path.exists(phase1_ckpt):
                        if is_main_process():
                            print(
                                f"  >> 连续 {edit_collapse_count} epoch collapse，"
                                f"回滚到 Phase 1 checkpoint"
                            )
                        ckpt = torch.load(phase1_ckpt, map_location=device)
                        raw_model.load_state_dict(ckpt["model_state_dict"])
                        phase2_initialized = False
                        edit_collapse_count = 0
            else:
                edit_collapse_count = 0

            rc_f1_thr = float(CONFIG.get("rc_f1_collapse_threshold", 0.0))
            grace_eps = int(CONFIG.get("phase2_collapse_grace_epochs", 0))
            epochs_in_phase2_for_grace = epoch - CONFIG["phase2_start_epoch"]
            rc_f1_collapsed = False
            if current_phase == 2 and epochs_in_phase2_for_grace >= grace_eps:
                cur_f1 = val_avg.get("rc_f1", 1.0)
                cur_neg = float(val_avg.get("rc_avg_prob_neg", 0.0))
                cur_pred = float(val_avg.get("rc_pred_count", 0.0))
                cur_gt = float(val_avg.get("rc_gt_count", 1.0))

                if rc_f1_thr > 0 and cur_f1 < rc_f1_thr:
                    if is_main_process():
                        print(
                            f"  !! [v20] 防崩 3: Phase 2 RC F1 collapse: ep{epoch} val_rc_f1="
                            f"{cur_f1:.4f} < {rc_f1_thr:.2f} → early stop"
                        )
                    rc_f1_collapsed = True

                neg_thr = float(CONFIG.get("rc_neg_prob_threshold", 0.0))
                neg_ratio_thr = float(CONFIG.get("rc_neg_prob_step_ratio", 0.0))
                if not rc_f1_collapsed and neg_thr > 0 and cur_neg > neg_thr:
                    if is_main_process():
                        print(
                            f"  !! [v20] 防崩 6: rc_avg_prob_neg ep{epoch}={cur_neg:.4f} "
                            f"> {neg_thr:.2f} 绝对阈值 → early stop"
                        )
                    rc_f1_collapsed = True
                if not rc_f1_collapsed and neg_ratio_thr > 0 and len(history) > 0:
                    prev_neg = float(history[-1].get("val_rc_avg_prob_neg", 0.0))
                    if prev_neg > 1e-6 and (cur_neg / prev_neg) > neg_ratio_thr:
                        if is_main_process():
                            print(
                                f"  !! [v20] 防崩 6: rc_avg_prob_neg ep{epoch}={cur_neg:.4f} "
                                f"vs ep{epoch-1}={prev_neg:.4f} ({cur_neg/prev_neg:.2f}x) "
                                f"> {neg_ratio_thr:.1f}x → early stop"
                            )
                        rc_f1_collapsed = True

                pred_ratio_thr = float(CONFIG.get("rc_pred_count_ratio_threshold", 0.0))
                if not rc_f1_collapsed and pred_ratio_thr > 0 and cur_gt > 1e-6:
                    pred_ratio = cur_pred / cur_gt
                    if pred_ratio > pred_ratio_thr:
                        if is_main_process():
                            print(
                                f"  !! [v20] 防崩 7: rc_pred_count/gt ep{epoch}="
                                f"{cur_pred:.2f}/{cur_gt:.2f}={pred_ratio:.2f}x "
                                f"> {pred_ratio_thr:.2f} → early stop"
                            )
                        rc_f1_collapsed = True

                edit_nz_thr = float(CONFIG.get("edit_nz_acc_collapse_threshold", 0.0))
                if not rc_f1_collapsed and edit_nz_thr > 0:

                    cur_edit_nz = float(val_avg.get("edit_recall", 1.0))
                    if cur_edit_nz < edit_nz_thr and len(history) > 0:
                        prev_edit_nz = float(history[-1].get("val_edit_recall", 1.0))
                        if prev_edit_nz < edit_nz_thr:

                            if is_main_process():
                                print(
                                    f"  !! [v20.2] 防崩 8: edit_nz_acc 连续 2 ep < {edit_nz_thr:.2f}; "
                                    f"ep{epoch-1}={prev_edit_nz:.4f}, ep{epoch}={cur_edit_nz:.4f} → early stop"
                                )
                            rc_f1_collapsed = True

        scheduler.step()

        lr_rc_floor = float(CONFIG.get("lr_rc_phase2_floor", 0.0))
        if lr_rc_floor > 0 and current_phase == 2 and len(optimizer.param_groups) > 1:
            if optimizer.param_groups[1]["lr"] < lr_rc_floor:
                optimizer.param_groups[1]["lr"] = lr_rc_floor

        if is_main_process():
            lrs = [pg["lr"] for pg in optimizer.param_groups]
            log_entry = {"epoch": epoch, "phase": current_phase}
            for k, v in train_avg.items():
                log_entry[f"train_{k}"] = v
            for k, v in val_avg.items():
                log_entry[f"val_{k}"] = v
            log_entry["lr_bert"] = lrs[0]
            log_entry["lr_rc"] = lrs[1]
            log_entry["lr_edit"] = lrs[2]
            if len(lrs) > 4:
                log_entry["lr_cross_frag"] = lrs[4]
            log_entry["grad_norm_bert"] = grad_norm_accum[0] / max(grad_norm_steps, 1)
            log_entry["grad_norm_rc"] = grad_norm_accum[1] / max(grad_norm_steps, 1)
            log_entry["grad_norm_edit"] = grad_norm_accum[2] / max(grad_norm_steps, 1)
            log_entry["edit_ramp"] = edit_ramp
            log_entry["ss_prob"] = ss_prob
            log_entry["nan_batches"] = nan_batch_total
            log_entry["spike_batches"] = spike_batch_total
            history.append(log_entry)
            pd.DataFrame(history).to_csv(history_path, index=False)

            save_metric_name = CONFIG.get("save_best_metric", "K_pos_main_mol_acc")
            best_metric = val_avg.get(save_metric_name, val_avg.get("mol_acc", 0.0))
            is_best = best_metric > best_mol_acc
            if is_best:
                best_mol_acc = best_metric

            is_best_rc_f1 = (current_phase == 1 and val_avg.get("rc_f1", 0) > best_rc_f1)
            if is_best_rc_f1:
                best_rc_f1 = val_avg["rc_f1"]

            save_path = f"{CONFIG['save_dir']}/model_epoch_{epoch}.pth"
            ckpt_data = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "contrastive_state_dict": contrastive_loss_fn.state_dict(),
                "best_mol_acc": best_mol_acc,
                "best_rc_f1": best_rc_f1,
                "phase2_initialized": phase2_initialized,
                "rc_temperature": CONFIG.get("rc_temperature", 1.0),
                "best_main_mol_acc": val_avg.get("main_mol_acc", 0.0),
                "strict_mol_acc": val_avg.get("mol_acc", 0.0),

                "best_K_pos_main_mol_acc": val_avg.get("K_pos_main_mol_acc", 0.0),
                "best_K4plus_main_mol_acc": val_avg.get("K4plus_main_mol_acc", 0.0),
                "best_bucket_avg_acc": val_avg.get("bucket_avg_acc", 0.0),
                "best_main_mol_acc_revised": val_avg.get("main_mol_acc_revised", 0.0),
            }
            torch.save(ckpt_data, save_path)
            if is_best:
                torch.save(ckpt_data, f"{CONFIG['save_dir']}/model_best.pth")
                print(f"  -> New best {save_metric_name}: {best_mol_acc:.1%}")
            if is_best_rc_f1:
                torch.save(ckpt_data, f"{CONFIG['save_dir']}/model_best_rc_f1.pth")
                print(f"  -> New best RC F1: {best_rc_f1:.1%} (Phase 1)")
            old_ckpt = f"{CONFIG['save_dir']}/model_epoch_{epoch - 3}.pth"
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

            es_patience = int(CONFIG.get("early_stop_patience", 0))
            es_metric_key = CONFIG.get("early_stop_metric", "val_K_pos_main_mol_acc").replace("val_", "")
            if es_patience > 0 and current_phase >= 2:
                cur_main_acc = float(val_avg.get(es_metric_key, 0.0))
                if cur_main_acc > best_main_acc_for_es:
                    best_main_acc_for_es = cur_main_acc
                    best_main_acc_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= es_patience:
                    if is_main_process():
                        print(
                            f"  !! [v23 §46] Early stop: best at ep{best_main_acc_epoch} "
                            f"{es_metric_key}={best_main_acc_for_es:.4f}, "
                            f"no improvement for {epochs_without_improvement} ep "
                            f"(patience={es_patience})"
                        )
                    break

            wd_K2 = float(CONFIG.get("watchdog_K2_threshold", 0.93))
            wd_K3 = float(CONFIG.get("watchdog_K3_threshold", 0.88))
            if current_phase >= 2 and epoch >= 2:
                cur_K2 = float(val_avg.get("K2_main_revised", 1.0))
                cur_K3 = float(val_avg.get("K3_main_revised", 1.0))
                if cur_K2 < wd_K2 or cur_K3 < wd_K3:
                    if is_main_process():
                        print(
                            f"  !! [v23 §46.7 watchdog] K2={cur_K2:.4f} (red {wd_K2}) "
                            f"K3={cur_K3:.4f} (red {wd_K3}) — Curriculum 退化警示"
                        )

            wd_grace = int(CONFIG.get("watchdog_K2_grace_epochs", 5))
            if (
                CONFIG.get("use_set_recall_loss", False)
                and current_phase >= 2
                and epoch >= wd_grace
                and val_avg.get("K2_mol_acc", 1.0) < (
                    CONFIG.get("watchdog_K2_baseline", 0.924)
                    - CONFIG.get("watchdog_K2_tolerance", 0.01)
                )
            ):
                if CONFIG["set_loss_k_weights"].get(2, 0.0) > 0:
                    CONFIG["set_loss_k_weights"][2] = 0.0
                    if is_main_process():
                        print(
                            f"  !! [v20.4] 防崩 12 触发: K=2 mol_acc {val_avg['K2_mol_acc']:.4f} "
                            f"< {CONFIG['watchdog_K2_baseline'] - CONFIG['watchdog_K2_tolerance']:.4f} "
                            f"→ set_loss_k_weights[2]=0.0 (永久关闭)"
                        )

        if is_ddp:
            best_tensor = torch.tensor(best_mol_acc, device=device)
            dist.broadcast(best_tensor, src=0)
            best_mol_acc = best_tensor.item()

        if locals().get("rc_f1_collapsed", False):
            if is_main_process():
                print(
                    f"[v20] Early stop at ep{epoch}: collapse threshold breached "
                    f"(防崩 3/6/7). 保留 model_phase1_final.pth, 跳出训练循环."
                )
            break

    cleanup_ddp()

if __name__ == "__main__":

    main()
