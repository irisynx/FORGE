"""推理侧化学约束解码 v0 - 评估驱动

用法:
  python tools/evaluate_constrained_decoding.py \
      --ckpt results_editgnn_v24/model_best.pth \
      --module train_forge \
      --output_dir analysis_reports/inference_constrained_decoding_v0 \
      --val_split  # 用 train_module data_dir 的 90/10 val
      [--data_dir_override <path>]
      [--N_list 0,1,2,3,4,6,10]
      [--batch_size 4]
      [--num_workers 2]
      [--limit_batches N]   # debug 用
      [--save_per_mol_records]  # 留 R/G 可视化用

输出:
  <output_dir>/main_table.json        - K-bucket × N 准确率
  <output_dir>/recovery_matrix.json   - per-N R/G 矩阵
  <output_dir>/per_mol_records.json   - 可选, 每分子记录 (含 GT smiles)
  <output_dir>/diag_summary.json      - 应用键数 / 修复 / H 吸收等
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from forge_decode_core import (
    DEFAULT_HYPERPARAMS,
    decode_batch,
    decode_batch_topk,
)

def compute_per_mol_correct(outputs, batch, T, edit_pred_override=None, enable_product_hook=False):
    """每分子返回 dict: K_bucket / correct_strict / correct_main_orig / correct_main_revised
       / correct_main_product / fail_rc / fail_edit / fail_coverage.

       enable_product_hook=True 时, 当 main_revised 失败但 v26 canonical SMILES
       匹配则 correct_main_product=True (对齐训练侧 K_pos_product_mol_acc).
    """
    device = outputs["edit_logits"].device
    batch_idx = batch.batch
    batch_size = int(batch_idx.max().item()) + 1
    y_delta_list = batch.y_delta_list
    smi_r_list = getattr(batch, "smi_r_list", [""] * batch_size)
    smi_p_list = getattr(batch, "smi_p_list", [""] * batch_size)
    product_match_fn = getattr(T, "v26_product_match_for_mol", None)

    if edit_pred_override is not None:
        edit_pred = edit_pred_override.to(device)
    else:
        edit_pred = outputs["edit_logits"].argmax(dim=-1)
    edit_labels = outputs["edit_labels"]
    rc_probs = outputs["rc_probs"]

    cand_edges_list = outputs.get("cand_edges", [])
    edge_counts = [e.size(0) for e in cand_edges_list]
    node_counts = torch.bincount(batch_idx, minlength=batch_size)
    node_offsets = torch.cat(
        [torch.tensor([0], device=device), node_counts.cumsum(0)[:-1]]
    )

    out_records = []
    offset = 0

    for mi in range(batch_size):
        rec = {
            "K": 0, "K_bucket": 0,
            "correct_strict": False,
            "correct_main_orig": False,
            "correct_main_revised": False,
            "correct_main_product": False,
            "fail_rc": False, "fail_edit": False, "fail_coverage": False,
        }

        k = int(edge_counts[mi]) if mi < len(edge_counts) else 0
        y_delta = y_delta_list[mi].to(device)
        gt_nonzero = (y_delta != 0).triu(diagonal=1)
        gt_edit_count = int(gt_nonzero.sum().item())

        rc_target_local = T.extract_rc_target(y_delta)
        K = int((rc_target_local > 0.5).sum().item())
        Kb = 0 if K == 0 else (2 if K == 2 else (3 if K == 3 else (4 if K == 4 else 5)))
        rec["K"] = K
        rec["K_bucket"] = Kb

        n_local = int(node_counts[mi].item())
        start_local = int(node_offsets[mi].item())
        edge_mask_local = (
            (batch.edge_index[0] >= start_local)
            & (batch.edge_index[0] < start_local + n_local)
        )
        local_ei = (batch.edge_index[:, edge_mask_local] - start_local).cpu()
        local_ea = batch.edge_attr[edge_mask_local].cpu().float()
        main_mask_local = T.compute_main_product_atoms(
            local_ei, local_ea, y_delta.cpu(), n_local,
        )

        if k == 0:
            if gt_edit_count == 0:
                rec["correct_strict"] = True
                rec["correct_main_orig"] = True
                rec["correct_main_revised"] = True
                rec["correct_main_product"] = True
            else:
                rec["fail_coverage"] = True
            out_records.append(rec)
            continue

        mol_pred = edit_pred[offset: offset + k]
        mol_true = edit_labels[offset: offset + k]
        cand_for_mol = cand_edges_list[mi] if mi < len(cand_edges_list) else None
        offset += k

        covered_edits = int((mol_true != 0).sum().item())
        if gt_edit_count != covered_edits:
            rec["fail_coverage"] = True
            out_records.append(rec)
            continue

        all_match = bool((mol_pred == mol_true).all().item())
        if all_match:
            rec["correct_strict"] = True
            rec["correct_main_orig"] = True
            rec["correct_main_revised"] = True
            rec["correct_main_product"] = True
            out_records.append(rec)
            continue

        start = int(node_offsets[mi].item())
        n = int(node_counts[mi].item())
        mol_rc_probs = rc_probs[start: start + n]
        mol_rc_pred = mol_rc_probs > 0.5
        rc_correct = bool((mol_rc_pred == (rc_target_local > 0.5)).all().item())
        if not rc_correct:
            rec["fail_rc"] = True
        else:
            rec["fail_edit"] = True

        cand_local = cand_for_mol.cpu()
        mask_main_or = (
            main_mask_local[cand_local[:, 0]] | main_mask_local[cand_local[:, 1]]
        )
        gt_edge_mask_cpu = (mol_true.cpu() != 0)
        both_in_main = (
            main_mask_local[cand_local[:, 0]] & main_mask_local[cand_local[:, 1]]
        )
        mask_main_revised = both_in_main | gt_edge_mask_cpu

        mol_pred_cpu = mol_pred.cpu()
        mol_true_cpu = mol_true.cpu()

        if mask_main_or.any():
            rec["correct_main_orig"] = bool(
                (mol_pred_cpu[mask_main_or] == mol_true_cpu[mask_main_or]).all().item()
            )
        else:
            rec["correct_main_orig"] = True

        if mask_main_revised.any():
            rec["correct_main_revised"] = bool(
                (mol_pred_cpu[mask_main_revised] == mol_true_cpu[mask_main_revised]).all().item()
            )
        else:
            rec["correct_main_revised"] = True

        if rec["correct_main_revised"]:
            rec["correct_main_product"] = True
        elif enable_product_hook and K > 0 and product_match_fn is not None:
            try:
                pl = product_match_fn(
                    smi_r_list[mi] if mi < len(smi_r_list) else "",
                    smi_p_list[mi] if mi < len(smi_p_list) else "",
                    cand_local, mol_pred_cpu, mol_true_cpu,
                )
            except Exception:
                pl = None
            rec["correct_main_product"] = bool(pl) if pl is not None else False
        else:
            rec["correct_main_product"] = False

        out_records.append(rec)

    return out_records

def compute_per_mol_topk(outputs, batch, T, edit_pred_list, enable_product_hook=False):
    """edit_pred_list: list 长度 K, 每个 = 一个 rank 的 full-batch edit_pred (rank 1..K).
       返回 per-mol dict: {K_bucket, topk{r->bool}}, topk[r] = ranks 1..r 内任一 product 对 (累积单调)."""
    K = len(edit_pred_list)
    rank_recs = [
        compute_per_mol_correct(outputs, batch, T, edit_pred_override=ep,
                                enable_product_hook=enable_product_hook)
        for ep in edit_pred_list
    ]
    n_mol = len(rank_recs[0])
    out = []
    for mi in range(n_mol):
        topk = {}
        cum = False
        for r in range(K):
            cum = cum or rank_recs[r][mi]["correct_main_product"]
            topk[r + 1] = cum
        out.append({"K_bucket": rank_recs[0][mi]["K_bucket"], "topk": topk})
    return out

def aggregate_topk_table(topk_records, K):
    """topk_records: list of {K_bucket, topk{1..K}}.
       返回 (table, stats): table[r] -> {K_pos, per_bucket{Kb->acc}} (product micro, 累积)."""
    buckets = (0, 2, 3, 4, 5)
    stats = {Kb: {"count": 0, **{r: 0 for r in range(1, K + 1)}} for Kb in buckets}
    for rec in topk_records:
        Kb = rec["K_bucket"]
        stats[Kb]["count"] += 1
        for r in range(1, K + 1):
            if rec["topk"][r]:
                stats[Kb][r] += 1
    table = {}
    for r in range(1, K + 1):
        num = sum(stats[Kb][r] for Kb in buckets if Kb != 0)
        den = sum(stats[Kb]["count"] for Kb in buckets if Kb != 0)
        num_all = sum(stats[Kb][r] for Kb in buckets)
        den_all = sum(stats[Kb]["count"] for Kb in buckets)
        table[r] = {
            "K_pos": num / den if den else 0.0,
            "overall": num_all / den_all if den_all else 0.0,
            "per_bucket": {
                Kb: (stats[Kb][r] / stats[Kb]["count"] if stats[Kb]["count"] else 0.0)
                for Kb in buckets
            },
        }
    return table, stats

def aggregate_main_table(records_per_N):
    """records_per_N: dict[N] -> list of per-mol records.
       返回 dict[N] -> {K_bucket -> {'correct_strict':_, 'correct_main_revised':_, 'count':_}}.
    """
    out = {}
    for N, recs in records_per_N.items():
        bucket_stats = {Kb: {"strict": 0, "main_orig": 0, "main_revised": 0, "main_product": 0, "count": 0}
                        for Kb in (0, 2, 3, 4, 5)}
        for r in recs:
            Kb = r["K_bucket"]
            bucket_stats[Kb]["count"] += 1
            if r["correct_strict"]:
                bucket_stats[Kb]["strict"] += 1
            if r["correct_main_orig"]:
                bucket_stats[Kb]["main_orig"] += 1
            if r["correct_main_revised"]:
                bucket_stats[Kb]["main_revised"] += 1
            if r.get("correct_main_product", False):
                bucket_stats[Kb]["main_product"] += 1

        total = sum(b["count"] for b in bucket_stats.values())
        K_pos_correct = sum(b["main_revised"] for Kb, b in bucket_stats.items() if Kb != 0)
        K_pos_count = sum(b["count"] for Kb, b in bucket_stats.items() if Kb != 0)
        K_pos_product_correct = sum(b["main_product"] for Kb, b in bucket_stats.items() if Kb != 0)
        K4plus_correct = sum(b["main_revised"] for Kb, b in bucket_stats.items() if Kb >= 4)
        K4plus_count = sum(b["count"] for Kb, b in bucket_stats.items() if Kb >= 4)
        K4plus_product = sum(b["main_product"] for Kb, b in bucket_stats.items() if Kb >= 4)

        bucket_acc = {}
        bucket_avg_terms = []
        bucket_avg_product_terms = []
        for Kb in (0, 2, 3, 4, 5):
            c = bucket_stats[Kb]["count"]
            acc = bucket_stats[Kb]["main_revised"] / c if c > 0 else 0.0
            acc_p = bucket_stats[Kb]["main_product"] / c if c > 0 else 0.0
            bucket_acc[f"K{Kb}_main_revised"] = acc
            bucket_acc[f"K{Kb}_main_product"] = acc_p
            if Kb != 0 and c > 0:
                bucket_avg_terms.append(acc)
                bucket_avg_product_terms.append(acc_p)

        out[N] = {
            "total": total,
            "K_pos_main_revised": K_pos_correct / K_pos_count if K_pos_count > 0 else 0.0,
            "K_pos_product_mol_acc": K_pos_product_correct / K_pos_count if K_pos_count > 0 else 0.0,
            "K_pos_count": K_pos_count,
            "K4plus_main_revised": K4plus_correct / K4plus_count if K4plus_count > 0 else 0.0,
            "K4plus_main_product": K4plus_product / K4plus_count if K4plus_count > 0 else 0.0,
            "K4plus_count": K4plus_count,
            "bucket_avg": float(np.mean(bucket_avg_terms)) if bucket_avg_terms else 0.0,
            "bucket_avg_product": float(np.mean(bucket_avg_product_terms)) if bucket_avg_product_terms else 0.0,
            "main_revised_overall": sum(b["main_revised"] for b in bucket_stats.values()) / total if total > 0 else 0.0,
            "main_product_overall": sum(b["main_product"] for b in bucket_stats.values()) / total if total > 0 else 0.0,
            **bucket_acc,
            "by_bucket": bucket_stats,
        }
    return out

def per_mol_ensemble_dprime(outputs, batch, edit_pred_argmax, edit_pred_decoder, tau_route):
    """§63.2.4(d) D': per-mol 路由 argmax 与 MolDecoder.
    高 RC conf (mean|rc_prob-0.5| >= τ) → MolDecoder; 低 conf → argmax.
    返回与 edit_pred_argmax 同 shape 的 routed pred + per-mol routing decision list.
    """
    device = edit_pred_argmax.device
    batch_idx = batch.batch
    batch_size = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
    rc_probs = outputs["rc_probs"]
    cand_edges_list = outputs.get("cand_edges", [])

    edit_pred = edit_pred_argmax.clone()
    routing_decisions = []
    edge_offset = 0
    for mi in range(batch_size):
        k = int(cand_edges_list[mi].size(0)) if mi < len(cand_edges_list) else 0
        mol_atom_mask = (batch_idx == mi)
        if mol_atom_mask.sum().item() == 0 or k == 0:
            routing_decisions.append('A')
            edge_offset += k
            continue
        mol_rc = rc_probs[mol_atom_mask]
        mol_conf = (mol_rc.float() - 0.5).abs().mean().item()
        if mol_conf >= tau_route:
            edit_pred[edge_offset:edge_offset + k] = edit_pred_decoder[edge_offset:edge_offset + k]
            routing_decisions.append('D')
        else:
            routing_decisions.append('A')
        edge_offset += k
    return edit_pred, routing_decisions

def aggregate_recovery_matrix(records_per_N, baseline_N=0):
    """对每个非 baseline N, 计算 R/G 矩阵 (按 K-bucket 和总).
       返回 dict[N] -> {K_bucket -> {recovered, regressed, both_correct, both_wrong}}.
    """
    base = records_per_N[baseline_N]
    out = {}
    for N, recs in records_per_N.items():
        if N == baseline_N:
            continue
        assert len(recs) == len(base), f"len mismatch N={N}"
        per_bucket = {Kb: {"recovered": 0, "regressed": 0, "both_correct": 0, "both_wrong": 0, "count": 0}
                      for Kb in (0, 2, 3, 4, 5)}
        total = {"recovered": 0, "regressed": 0, "both_correct": 0, "both_wrong": 0, "count": 0}

        for r0, rN in zip(base, recs):
            assert r0["K_bucket"] == rN["K_bucket"], "K bucket mismatch — record order broken"
            Kb = r0["K_bucket"]
            c0 = r0["correct_main_revised"]
            cN = rN["correct_main_revised"]
            cell = "both_correct" if (c0 and cN) else                   "recovered" if (not c0 and cN) else                   "regressed" if (c0 and not cN) else "both_wrong"
            per_bucket[Kb][cell] += 1
            per_bucket[Kb]["count"] += 1
            total[cell] += 1
            total["count"] += 1

        per_bucket_pp = {}
        for Kb in (0, 2, 3, 4, 5):
            c = per_bucket[Kb]["count"]
            net = per_bucket[Kb]["recovered"] - per_bucket[Kb]["regressed"]
            per_bucket_pp[f"K{Kb}_net_gain_pp"] = (net / c * 100) if c > 0 else 0.0

        net_total = total["recovered"] - total["regressed"]
        out[N] = {
            "total": total,
            "by_bucket": per_bucket,
            "net_gain_pp_overall": (net_total / total["count"] * 100) if total["count"] > 0 else 0.0,
            "RG_ratio": (total["recovered"] / total["regressed"]) if total["regressed"] > 0 else float("inf"),
            **per_bucket_pp,
        }
    return out

def aggregate_diag(diag_per_N):
    out = {}
    for N, diags in diag_per_N.items():
        if not diags:
            out[N] = {}
            continue
        total = len(diags)
        out[N] = {
            "n_mols": total,
            "avg_n_apply": float(np.mean([d.get("n_apply", 0) for d in diags])),
            "avg_n_repair": float(np.mean([d.get("n_repair", 0) for d in diags])),
            "avg_n_h_absorb": float(np.mean([d.get("n_h_absorb", 0) for d in diags])),
            "avg_n_rollback": float(np.mean([d.get("n_rollback", 0) for d in diags])),
            "frac_h_absorbed": float(np.mean([1 if d.get("n_h_absorb", 0) > 0 else 0 for d in diags])),
            "frac_rollback": float(np.mean([1 if d.get("n_rollback", 0) > 0 else 0 for d in diags])),
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module", required=True, help="e.g. train_forge")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--test_set", default=None)
    ap.add_argument("--val_split", action="store_true")
    ap.add_argument("--data_dir_override", default=None)
    ap.add_argument("--N_list", default="0,1,2,3,4,6,10")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit_batches", type=int, default=-1)
    ap.add_argument("--save_per_mol_records", action="store_true")
    ap.add_argument("--tau_apply", type=float, default=DEFAULT_HYPERPARAMS["tau_apply"])
    ap.add_argument("--tau_repair", type=float, default=DEFAULT_HYPERPARAMS["tau_repair"])
    ap.add_argument("--tau_stop", type=float, default=DEFAULT_HYPERPARAMS["tau_stop"])
    ap.add_argument("--enable_product_hook", action="store_true",
                    help="启用 v26 product canonical SMILES hook (对齐训练侧 K_pos_product_mol_acc)")
    ap.add_argument("--beam_size", type=int, default=1,
                    help="MolDecoder beam search size (§63.2.3 E). 1=greedy (default, back-compat). "
                         "B>=2 启用 beam, 预期 +0.5-1pp, 推理慢 ~B^2/B x")
    ap.add_argument("--ensemble_routing", default="none", choices=["none", "rc_conf"],
                    help="§63.2.4(d) D' per-mol ensemble: argmax (N=0) vs MolDecoder (--ensemble_base_N). "
                         "rc_conf 按 mol RC 置信度路由 (高 conf → MolDecoder, 低 → argmax). "
                         "需 N_list 同时含 0 和 ensemble_base_N. 输出附加 pseudo-N=-1 行 (D'(N=B,τ=T))")
    ap.add_argument("--ensemble_base_N", type=int, default=3,
                    help="D' 用作 'decoder branch' 的 MolDecoder N (默认 3, 即 sweet spot)")
    ap.add_argument("--ensemble_route_tau", type=float, default=0.30,
                    help="D' rc_conf 路由阈值 (mean|rc_prob-0.5| >= τ → 用 decoder, else argmax)")
    ap.add_argument("--topk", type=int, default=1,
                    help="Top-K 评测 (beam-based). 1=关闭, 走原 N_list 路径. >1 时 rank-1=greedy"
                         "(N=topk_N) baseline, ranks 2..K=beam 其余互异解, 输出 Top-K 表 (product micro 累积).")
    ap.add_argument("--topk_beam", type=int, default=0,
                    help="top-K 的 beam 宽度 B (0=自动 max(2K,8)). 应 >= topk 以抵 cls_step 塌解.")
    ap.add_argument("--topk_N", type=int, default=3,
                    help="top-K decode 的 max_loops (anchor greedy 用; 默认 3 = sweet spot).")
    args = ap.parse_args()

    if not args.test_set and not args.val_split:
        print("[fatal] 必须指定 --test_set 或 --val_split"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    N_list = [int(x) for x in args.N_list.split(",")]
    print(f"[N_list] {N_list}")
    print(f"[beam_size] {args.beam_size} {'(greedy)' if args.beam_size == 1 else f'(beam B={args.beam_size})'}")

    use_dprime = args.ensemble_routing == "rc_conf"
    if use_dprime:
        assert 0 in N_list and args.ensemble_base_N in N_list,            f"--ensemble_routing rc_conf 需 N_list 同时含 0 和 {args.ensemble_base_N}, 当前 {N_list}"
        print(f"[ensemble D'] routing=rc_conf, base_N={args.ensemble_base_N}, "
              f"τ={args.ensemble_route_tau:.2f} (mean|rc_prob-0.5| >= τ → MolDecoder)")

    hyperparams = {
        "tau_apply": args.tau_apply,
        "tau_repair": args.tau_repair,
        "tau_stop": args.tau_stop,
        "max_inner_iter": DEFAULT_HYPERPARAMS["max_inner_iter"],
    }
    print(f"[hyperparams] {hyperparams}")

    T = importlib.import_module(args.module)
    T.CONFIG["use_compile"] = False

    pretrained = T.load_pretrained_model(T.CONFIG["elec_ckpt_path"], device)
    model = T.EditGNNModel(pretrained, T.CONFIG).to(device)
    model.eval()

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] {args.ckpt} loaded (missing={len(missing)} unexpected={len(unexpected)})")
    ep = state.get("epoch", "?") if isinstance(state, dict) else "?"
    print(f"[ckpt] epoch={ep}")

    if args.test_set:
        ds = T.DiskDataset(args.test_set, augment=False, data_ratio=1.0)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=T.editgnn_collate,
            pin_memory=True, persistent_workers=False,
        )
        loader_label = f"test_set:{args.test_set}"
    else:
        data_dir = args.data_dir_override or T.CONFIG["data_dir"]
        full = T.DiskDataset(data_dir, augment=False, data_ratio=1.0)
        g = torch.Generator().manual_seed(42)
        train_size = int(0.9 * len(full))
        _, val_ds = torch.utils.data.random_split(
            full, [train_size, len(full) - train_size], generator=g,
        )
        loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=T.editgnn_collate,
            pin_memory=True,
        )
        loader_label = f"val_split:{data_dir}"
    print(f"[loader] {loader_label}")

    use_rc_refine = T.CONFIG.get("rc_refine_in_eval", True)

    topk_records = []
    topk_distinct = []
    topk_beam = args.topk_beam if args.topk_beam > 0 else max(2 * args.topk, 8)
    if args.topk > 1:
        print(f"[topk] K={args.topk}, beam B={topk_beam}, anchor N(max_loops)={args.topk_N} "
              f"(rank-1=greedy baseline, ranks 2..K=beam rerank-by-full-LL)")

    records_per_N = {N: [] for N in N_list}
    diag_per_N = {N: [] for N in N_list if N > 0}
    DPRIME_KEY = -1
    if use_dprime:
        records_per_N[DPRIME_KEY] = []
        dprime_routing_counts = {'D': 0, 'A': 0}

    t0 = time.time()
    n_b = 0
    n_mol = 0
    decode_time = {N: 0.0 for N in N_list if N > 0}
    forward_time = 0.0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)

            t1 = time.time()
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch, training=False, phase=2, rc_refine=use_rc_refine)
            except Exception as e:
                print(f"  [forward fail] batch {n_b}: {type(e).__name__}: {e}")
                continue
            forward_time += time.time() - t1

            if outputs["edit_logits"].size(0) == 0:
                continue

            if args.topk > 1:
                t2 = time.time()
                edit_pred_list, tk_diag = decode_batch_topk(
                    outputs, batch, T, hyperparams,
                    max_loops=args.topk_N, B=topk_beam, K=args.topk,
                )
                topk_records.extend(compute_per_mol_topk(
                    outputs, batch, T, edit_pred_list,
                    enable_product_hook=args.enable_product_hook,
                ))
                topk_distinct.extend([d["n_distinct"] for d in tk_diag if d.get("k", 0) > 0])
                n_b += 1
                n_mol = len(topk_records)
                if n_b % 50 == 0:
                    tk_now, _ = aggregate_topk_table(topk_records, args.topk)
                    md = float(np.mean(topk_distinct)) if topk_distinct else 0.0
                    el = (time.time() - t0) / 60
                    print(f"  [{n_b}] mols={n_mol:>7} | top1={tk_now[1]['K_pos']:.4f} "
                          f"top{args.topk}={tk_now[args.topk]['K_pos']:.4f} | "
                          f"distinct={md:.2f} | {el:.1f}min")
                if args.limit_batches > 0 and n_b >= args.limit_batches:
                    print(f"[limit] stopped at batch {n_b}")
                    break
                continue

            preds_per_N = {}
            for N in N_list:
                t2 = time.time()
                if N == 0:
                    edit_pred = outputs["edit_logits"].argmax(dim=-1)
                    diags = None
                else:
                    edit_pred, diags = decode_batch(outputs, batch, T, hyperparams,
                                                   max_loops=N, beam_size=args.beam_size)
                    diag_per_N[N].extend(diags)
                if N > 0:
                    decode_time[N] += time.time() - t2
                preds_per_N[N] = edit_pred

                recs = compute_per_mol_correct(outputs, batch, T, edit_pred_override=edit_pred,
                                               enable_product_hook=args.enable_product_hook)
                records_per_N[N].extend(recs)

            if use_dprime:
                edit_pred_dp, routing = per_mol_ensemble_dprime(
                    outputs, batch,
                    preds_per_N[0], preds_per_N[args.ensemble_base_N],
                    args.ensemble_route_tau,
                )
                for r in routing:
                    dprime_routing_counts[r] += 1
                recs_dp = compute_per_mol_correct(outputs, batch, T, edit_pred_override=edit_pred_dp,
                                                  enable_product_hook=args.enable_product_hook)
                records_per_N[DPRIME_KEY].extend(recs_dp)

            n_b += 1
            n_mol = len(records_per_N[N_list[0]])

            if n_b % 50 == 0:
                el = (time.time() - t0) / 60
                main_now = aggregate_main_table({N: records_per_N[N] for N in [0, N_list[-1]]})
                prod_tag = " (prod)" if args.enable_product_hook else ""
                print(f"  [{n_b}] mols={n_mol:>7} | N=0 rev={main_now[0]['K_pos_main_revised']:.4f}"
                      f"{prod_tag if args.enable_product_hook else ''} "
                      f"prod={main_now[0]['K_pos_product_mol_acc']:.4f} | "
                      f"N={N_list[-1]} rev={main_now[N_list[-1]]['K_pos_main_revised']:.4f} "
                      f"prod={main_now[N_list[-1]]['K_pos_product_mol_acc']:.4f} | {el:.1f}min "
                      f"(fwd={forward_time/60:.1f} dec_max={max(decode_time.values())/60:.1f})")

            if args.limit_batches > 0 and n_b >= args.limit_batches:
                print(f"[limit] stopped at batch {n_b}")
                break

    print(f"\n[done] {n_b} batches, {n_mol} mols, {(time.time()-t0)/60:.1f}min")
    print(f"  forward: {forward_time/60:.1f}min")
    for N in N_list:
        if N > 0:
            print(f"  decode N={N}: {decode_time[N]/60:.1f}min")

    if args.topk > 1:
        topk_table, topk_stats = aggregate_topk_table(topk_records, args.topk)
        mean_distinct = float(np.mean(topk_distinct)) if topk_distinct else 0.0
        prod_tag = " (product hook ON)" if args.enable_product_hook else " (main_revised only)"
        print(f"\n=== Top-K Table (product micro by rank){prod_tag} ===")
        print(f"{'rank':>6} | {'overall':>8} {'K_pos':>8} | "
              f"{'K=0':>7} {'K=2':>7} {'K=3':>7} {'K=4':>7} {'K=5+':>7}")
        for r in range(1, args.topk + 1):
            t = topk_table[r]
            pb = t["per_bucket"]
            print(f"{('top-'+str(r)):>6} | {t['overall']:.4f}   {t['K_pos']:.4f} | "
                  f"{pb[0]:.4f}  {pb[2]:.4f}  {pb[3]:.4f}  {pb[4]:.4f}  {pb[5]:.4f}")
        cnts = {Kb: topk_stats[Kb]["count"] for Kb in (0, 2, 3, 4, 5)}
        tot = sum(cnts.values())
        print("  [counts] " + " ".join(
            f"K={kb}:{cnts[kb]}({100*cnts[kb]/tot:.1f}%)" for kb in (0, 2, 3, 4, 5)) +
            f"  total={tot}")
        print(f"\n[distinct] 平均互异解数 = {mean_distinct:.2f} (>1.5 才说明 top-K 有多样性)")
        print(f"[sanity]   top-1 K_pos = {topk_table[1]['K_pos']:.4f} "
              f"overall = {topk_table[1]['overall']:.4f} (top-1 应 == 当前 greedy baseline)")
        out_tk = {
            "ckpt": args.ckpt, "module": args.module, "loader": loader_label,
            "topk": args.topk, "topk_beam": topk_beam, "topk_N": args.topk_N,
            "hyperparams": hyperparams, "n_mols": n_mol, "n_batches": n_b,
            "enable_product_hook": args.enable_product_hook,
            "mean_distinct": mean_distinct,
            "topk_table": {r: topk_table[r] for r in range(1, args.topk + 1)},
            "by_bucket_counts": {str(Kb): topk_stats[Kb] for Kb in topk_stats},
        }
        tk_path = os.path.join(args.output_dir, "topk_table.json")
        with open(tk_path, "w") as f:
            json.dump(out_tk, f, indent=2, default=str)
        print(f"\nSaved: {tk_path}")
        return

    main_table = aggregate_main_table(records_per_N)
    recovery = aggregate_recovery_matrix(records_per_N, baseline_N=0)
    diag_summary = aggregate_diag(diag_per_N)

    print_order = list(N_list) + ([DPRIME_KEY] if use_dprime else [])

    def _N_label(n):
        return f"D'(N={args.ensemble_base_N},τ={args.ensemble_route_tau:.2f})" if n == DPRIME_KEY else str(n)

    print("\n=== Main Table (K_pos_main_revised by N) ===")
    print(f"{'N':>22} | {'K_pos':>8} {'K=2':>7} {'K=3':>7} {'K=4':>7} {'K=5+':>7} | {'bucket_avg':>10}")
    for N in print_order:
        m = main_table[N]
        print(f"{_N_label(N):>22} | {m['K_pos_main_revised']:.4f} | "
              f"{m['K2_main_revised']:.4f} {m['K3_main_revised']:.4f} "
              f"{m['K4_main_revised']:.4f} {m['K5_main_revised']:.4f} | "
              f"{m['bucket_avg']:.4f}")

    if args.enable_product_hook:
        print("\n=== Main Table (K_pos_product_mol_acc by N, 对齐训练侧) ===")
        print(f"{'N':>22} | {'K_pos':>8} {'K=2':>7} {'K=3':>7} {'K=4':>7} {'K=5+':>7} | {'bucket_avg':>10}")
        for N in print_order:
            m = main_table[N]
            print(f"{_N_label(N):>22} | {m['K_pos_product_mol_acc']:.4f} | "
                  f"{m['K2_main_product']:.4f} {m['K3_main_product']:.4f} "
                  f"{m['K4_main_product']:.4f} {m['K5_main_product']:.4f} | "
                  f"{m['bucket_avg_product']:.4f}")

    if use_dprime:
        total_dec = dprime_routing_counts['D'] + dprime_routing_counts['A']
        if total_dec > 0:
            print(f"\n[D' routing] D (decoder) = {dprime_routing_counts['D']:>6} "
                  f"({100*dprime_routing_counts['D']/total_dec:.1f}%) | "
                  f"A (argmax) = {dprime_routing_counts['A']:>6} "
                  f"({100*dprime_routing_counts['A']/total_dec:.1f}%)")

    print("\n=== Recovery Matrix (vs baseline N=0) ===")
    print(f"{'N':>22} | {'R':>6} {'G':>6} {'R/G':>6} {'NetGain%':>8} | {'K2_pp':>6} {'K3_pp':>6} {'K4_pp':>6} {'K5+_pp':>6}")
    for N in print_order:
        if N == 0:
            continue
        r = recovery[N]
        rg = r["RG_ratio"]
        rg_str = f"{rg:.2f}" if rg != float("inf") else "inf"
        print(f"{_N_label(N):>22} | {r['total']['recovered']:>6} {r['total']['regressed']:>6} {rg_str:>6} "
              f"{r['net_gain_pp_overall']:>8.2f} | "
              f"{r['K2_net_gain_pp']:>6.2f} {r['K3_net_gain_pp']:>6.2f} "
              f"{r['K4_net_gain_pp']:>6.2f} {r['K5_net_gain_pp']:>6.2f}")

    print("\n=== Diag Summary ===")
    for N in [N for N in N_list if N > 0]:
        d = diag_summary[N]
        print(f"  N={N}: avg_apply={d['avg_n_apply']:.2f} repair={d['avg_n_repair']:.2f} "
              f"h_absorb={d['avg_n_h_absorb']:.2f} rollback={d['avg_n_rollback']:.2f} "
              f"frac_h_absorbed={d['frac_h_absorbed']:.3f} frac_rollback={d['frac_rollback']:.3f}")

    out = {
        "ckpt": args.ckpt,
        "module": args.module,
        "loader": loader_label,
        "N_list": N_list,
        "hyperparams": hyperparams,
        "n_mols": n_mol,
        "n_batches": n_b,
        "elapsed_min": (time.time() - t0) / 60,
        "main_table": main_table,
        "recovery_matrix": recovery,
        "diag_summary": diag_summary,
    }
    main_path = os.path.join(args.output_dir, "main_table.json")
    rec_path = os.path.join(args.output_dir, "recovery_matrix.json")
    diag_path = os.path.join(args.output_dir, "diag_summary.json")
    full_path = os.path.join(args.output_dir, "results_full.json")
    with open(main_path, "w") as f:
        json.dump(main_table, f, indent=2, default=str)
    with open(rec_path, "w") as f:
        json.dump(recovery, f, indent=2, default=str)
    with open(diag_path, "w") as f:
        json.dump(diag_summary, f, indent=2, default=str)
    with open(full_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {main_path}\n       {rec_path}\n       {diag_path}\n       {full_path}")

    if args.save_per_mol_records:
        per_mol_path = os.path.join(args.output_dir, "per_mol_records.json")
        records_serial = {str(N): records_per_N[N] for N in N_list}
        with open(per_mol_path, "w") as f:
            json.dump(records_serial, f, default=str)
        print(f"Saved per-mol records: {per_mol_path}")

if __name__ == "__main__":
    main()
