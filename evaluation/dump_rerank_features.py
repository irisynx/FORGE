"""路线3 Step1 (FULL 版) — K=0 fallback, 报全集 39,768 mol 数字, 不再 skip K=0.

差异 vs dump_rerank_features.py:
  1) K=0 mol 不再 `continue` skip; 写 1 行 placeholder feat (全零 + rank=0), correct 字段直接取
     `rank_recs[0][mi]["correct_main_product"]`(K=0 等价 "predict no change"). 这样 reranker apply
     时 K=0 mol 只有一个候选可选 → greedy/handcraft/learned/oracle 四口径都退化为 base 决策, 不
     歧义不漏统计.
  2) 默认 --test_mod=1 --test_lt=1 → 所有 mol 标 split=1, eval_reranker_topk 直接在全集上 apply
     已训好的 reranker_mlp.pt(不 retrain).
  3) 输出 npz schema 与原 dump 完全一致, 可直接喂 tools/train_eval_reranker.py.

用法 (USPTO test 全 39,768):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python dump_rerank_features.py \\
      --ckpt results_editgnn_v34_E1_uspto_v3/model_best.pth \\
      --module train_forge_uspto \\
      --test_set uspto_480k_rxnmapped/test \\
      --out analysis_reports/rerank_feats_v3_uspto_FULL/feats.npz \\
      --topk 5 --topk_beam 8 --topk_N 3 --batch_size 4
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from forge_decode_core import DEFAULT_HYPERPARAMS, decode_batch_topk
from evaluate_decoding import compute_per_mol_correct
from forge_product_rerank import plausibility_score

FEAT_NAMES = [
    "ll", "ll_gap", "rank", "n_edit", "frac_break", "frac_form",
    "mean_conf", "min_conf", "conf_entropy",
    "plaus", "sanitizable", "n_radical", "n_frag",
    "mw_prod", "mw_gap_r", "ring_change", "heavy_gap_r",
    "rc_mean_on_edit", "rc_max_on_edit",
]
_CLS_BREAK = {2, 4, 6}
_CLS_FORM = {1, 3, 5}

def _rdkit_props(smi):
    if not smi:
        return None
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return (0.0, 0, 0, 0, smi.count(".") + 1, 0)
    san = 1
    try:
        Chem.SanitizeMol(m)
    except Exception:
        san = 0
    try:
        mw = float(Descriptors.MolWt(m))
    except Exception:
        mw = 0.0
    nring = m.GetRingInfo().NumRings()
    nheavy = m.GetNumHeavyAtoms()
    nrad = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    try:
        nfrag = len(Chem.GetMolFrags(m))
    except Exception:
        nfrag = 1
    return (mw, nring, nheavy, nrad, nfrag, san)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--module", required=True)
    ap.add_argument("--test_set", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--limit_batches", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--topk_beam", type=int, default=8)
    ap.add_argument("--topk_N", type=int, default=3)
    ap.add_argument("--test_mod", type=int, default=1,
                    help="FULL 版默认 1 → 全部标 split=1(全集评)")
    ap.add_argument("--test_lt", type=int, default=1)
    ap.add_argument("--ss_rc_threshold", type=float, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    hyperparams = {"tau_apply": 0.40, "tau_repair": 0.25, "tau_stop": 0.30,
                   "max_inner_iter": DEFAULT_HYPERPARAMS["max_inner_iter"]}

    T = importlib.import_module(args.module)
    T.CONFIG["use_compile"] = False
    canon_fn = T._v26_product_canonical
    CLS2DELTA = T._CLS_TO_DELTA_INT

    pretrained = T.load_pretrained_model(T.CONFIG["elec_ckpt_path"], device)
    model = T.EditGNNModel(pretrained, T.CONFIG).to(device)
    model.eval()
    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print("[ckpt] %s ep=%s" % (args.ckpt, state.get("epoch", "?")))
    _ckpt_thr = state.get("ss_rc_threshold") if isinstance(state, dict) else None
    if args.ss_rc_threshold is not None:
        T.CONFIG["ss_rc_threshold"] = args.ss_rc_threshold
        print("[rc_thr] CLI ss_rc_threshold=%s" % args.ss_rc_threshold)
    elif _ckpt_thr is not None:
        T.CONFIG["ss_rc_threshold"] = float(_ckpt_thr)
        print("[rc_thr] ckpt ss_rc_threshold=%.3f" % float(_ckpt_thr))
    else:
        print("[rc_thr] WARN: 用 CONFIG 默认 %s" % T.CONFIG.get("ss_rc_threshold"))
    _ct = state.get("rc_temperature") if isinstance(state, dict) else None
    if _ct is not None: T.CONFIG["rc_temperature"] = float(_ct)

    ds = T.DiskDataset(args.test_set, augment=False, data_ratio=1.0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=T.editgnn_collate, pin_memory=True)
    print("[loader] %s (chunks=%d)" % (args.test_set, len(ds)))
    use_rc_refine = T.CONFIG.get("rc_refine_in_eval", True)
    K = args.topk
    n_feat = len(FEAT_NAMES)

    rows_X, rows_mol, rows_rank, rows_corr, rows_Kb, rows_split = [], [], [], [], [], []
    mol_gid = 0
    k0_correct = 0
    k0_total = 0
    t0 = time.time()
    n_b = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = batch.to(device)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(batch, training=False, phase=2, rc_refine=use_rc_refine)
            except Exception as e:
                print("  [fwd fail] b%d %s" % (n_b, e)); continue

            cand_edges_list = outputs.get("cand_edges", [])
            edge_counts = [e.size(0) for e in cand_edges_list]
            is_test = (n_b % args.test_mod) < args.test_lt

            has_any_edit = outputs["edit_logits"].size(0) > 0
            if has_any_edit:
                edit_pred_list, _ = decode_batch_topk(outputs, batch, T, hyperparams,
                                                      max_loops=args.topk_N, B=args.topk_beam, K=K)
                edit_pred_list = [ep.to(device) if isinstance(ep, torch.Tensor)
                                  else torch.as_tensor(ep, device=device) for ep in edit_pred_list]
                rank_recs = [compute_per_mol_correct(outputs, batch, T, edit_pred_override=ep,
                                                     enable_product_hook=True) for ep in edit_pred_list]
                n_mol_b = len(rank_recs[0])

                edit_logits = outputs["edit_logits"].float()
                log_probs = torch.log_softmax(edit_logits, dim=-1).cpu().numpy()
                probs = torch.softmax(edit_logits, dim=-1).cpu().numpy()
                rc_probs = outputs["rc_probs"].float().cpu().numpy()
                batch_idx = batch.batch.cpu().numpy()
                node_counts = np.bincount(batch_idx, minlength=n_mol_b)
                node_offsets = np.concatenate([[0], np.cumsum(node_counts)[:-1]])
                smi_r_list = getattr(batch, "smi_r_list", [""] * n_mol_b)
                ep_cpu = [ep.cpu().numpy() for ep in edit_pred_list]
            else:

                rank_recs = [compute_per_mol_correct(outputs, batch, T,
                                                     edit_pred_override=torch.zeros(0, dtype=torch.long, device=device),
                                                     enable_product_hook=True)]
                n_mol_b = len(rank_recs[0])

            offset = 0
            for mi in range(n_mol_b):
                k = int(edge_counts[mi]) if mi < len(edge_counts) else 0
                Kb = rank_recs[0][mi]["K_bucket"]
                gid = mol_gid; mol_gid += 1
                if k == 0:
                    k0_total += 1
                    correct0 = int(bool(rank_recs[0][mi]["correct_main_product"]))
                    k0_correct += correct0
                    rows_X.append([0.0] * n_feat)
                    rows_mol.append(gid); rows_rank.append(0)
                    rows_corr.append(correct0)
                    rows_Kb.append(Kb); rows_split.append(1 if is_test else 0)
                    continue
                smi_r = smi_r_list[mi] if mi < len(smi_r_list) else ""
                rprops = _rdkit_props(smi_r)
                r_mw = rprops[0] if rprops else 0.0
                r_ring = rprops[1] if rprops else 0
                r_heavy = rprops[2] if rprops else 0
                cand_local = cand_edges_list[mi].cpu().numpy()
                n_loc = int(node_counts[mi]); start_loc = int(node_offsets[mi])
                rc_loc = rc_probs[start_loc:start_loc + n_loc]

                lp_mol = log_probs[offset:offset + k]
                p_mol = probs[offset:offset + k]
                ll_r1 = None
                for r in range(K):
                    cls = ep_cpu[r][offset:offset + k]
                    ll = float(lp_mol[np.arange(k), cls].sum())
                    if r == 0:
                        ll_r1 = ll
                    nz = cls != 0
                    n_edit = int(nz.sum())
                    if n_edit > 0:
                        conf = p_mol[np.arange(k), cls][nz]
                        mean_conf = float(conf.mean()); min_conf = float(conf.min())
                        ent = float(-(conf * np.log(conf + 1e-9)).mean())
                        frac_break = float(np.isin(cls[nz], list(_CLS_BREAK)).mean())
                        frac_form = float(np.isin(cls[nz], list(_CLS_FORM)).mean())
                        edit_atoms = np.unique(cand_local[nz][:, :2].reshape(-1))
                        edit_atoms = edit_atoms[edit_atoms < n_loc]
                        rc_e = rc_loc[edit_atoms] if len(edit_atoms) else np.array([0.0])
                        rc_mean = float(rc_e.mean()); rc_max = float(rc_e.max())
                    else:
                        mean_conf = min_conf = ent = frac_break = frac_form = 0.0
                        rc_mean = rc_max = 0.0
                    changes = [(int(cand_local[e, 0]), int(cand_local[e, 1]), CLS2DELTA.get(int(cls[e]), 0))
                               for e in range(k) if cls[e] != 0]
                    try:
                        canon = canon_fn(smi_r, changes, None)
                    except Exception:
                        canon = ""
                    plaus = plausibility_score(canon)
                    pp = _rdkit_props(canon)
                    if pp:
                        mw_p, ring_p, heavy_p, nrad, nfrag, san = pp
                    else:
                        mw_p, ring_p, heavy_p, nrad, nfrag, san = 0.0, 0, 0, 0, 1, 0
                    feat = [
                        ll, ll - (ll_r1 if ll_r1 is not None else ll), float(r),
                        float(n_edit), frac_break, frac_form,
                        mean_conf, min_conf, ent,
                        plaus, float(san), float(nrad), float(nfrag),
                        mw_p, mw_p - r_mw, float(ring_p - r_ring), float(heavy_p - r_heavy),
                        rc_mean, rc_max,
                    ]
                    rows_X.append(feat); rows_mol.append(gid); rows_rank.append(r)
                    rows_corr.append(int(bool(rank_recs[r][mi]["correct_main_product"])))
                    rows_Kb.append(Kb); rows_split.append(1 if is_test else 0)
                offset += k

            n_b += 1
            if n_b % 50 == 0:
                print("  [b%d] mols=%d rows=%d k0=%d/%d (%.1fmin)"
                      % (n_b, mol_gid, len(rows_X), k0_correct, k0_total, (time.time() - t0) / 60))
            if args.limit_batches and n_b >= args.limit_batches:
                break

    X = np.asarray(rows_X, dtype=np.float32)
    np.savez_compressed(
        args.out, X=X, feat_names=np.array(FEAT_NAMES),
        mol_id=np.asarray(rows_mol, np.int64), rank=np.asarray(rows_rank, np.int8),
        correct=np.asarray(rows_corr, np.int8), K_bucket=np.asarray(rows_Kb, np.int8),
        split=np.asarray(rows_split, np.int8))
    n_test_mol = len(set(m for m, s in zip(rows_mol, rows_split) if s == 1))
    n_tr_mol = len(set(m for m, s in zip(rows_mol, rows_split) if s == 0))
    print("[done] rows=%d  feats=%d  mols(train=%d test=%d)  K=0:%d/%d  -> %s (%.1fmin)"
          % (len(rows_X), X.shape[1], n_tr_mol, n_test_mol, k0_correct, k0_total,
             args.out, (time.time() - t0) / 60))

if __name__ == "__main__":
    main()
