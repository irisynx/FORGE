"""Per-seed reranker MLP top-K evaluation for Fig 4h "mean ± s.d. over 3 seeds".

The existing `train_eval_reranker.py` trains N seeds and averages their scores
(ensemble), reporting a single top-K — there is no per-seed test number, so the
"mean ± s.d." figure caption cannot be sourced from its JSON.

This script trains the same MLP architecture / split / hyperparameters as
`train_eval_reranker.py` but evaluates EACH seed independently on the test set,
then aggregates mean / std across seeds. It also reports the ensemble number
(should match the existing `reranker_mlp_result.json::methods.learned.K_pos`).

Usage (remote):
    cd /data/chara/project/AI4S
    uv run python tools/eval_reranker_perseed.py \
        --feats analysis_reports/rerank_feats_v34_E1_top15/feats.npz \
        --out   analysis_reports/rerank_feats_v34_E1_top15/reranker_mlp_perseed.json \
        --seeds 0,1,2 --topk 1,3,5,10

USPTO:
    uv run python tools/eval_reranker_perseed.py \
        --feats analysis_reports/rerank_feats_v3_top15/feats.npz \
        --out   analysis_reports/rerank_feats_v3_top15/reranker_mlp_perseed.json \
        --seeds 0,1,2 --topk 1,3,5,10
"""
import argparse
import json
import os
import numpy as np
import torch

from train_eval_reranker import (
    MLP,
    group_by_mol,
    topk_acc_per_bucket,
    train_one_seed,
    _expand_score_to_rows,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="feats.npz from dump_rerank_features.py")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--seeds", default="0,1,2", help="comma list of seeds, e.g. 0,1,2 or 42,1337,2024")
    ap.add_argument("--topk", default="1,3,5,10", help="comma list of N for top-N evaluation")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    d = np.load(args.feats, allow_pickle=True)
    X = d["X"].astype(np.float32)
    fn = list(d["feat_names"])
    mol_id = d["mol_id"]; rank = d["rank"]; correct = d["correct"].astype(np.int64)
    Kb = d["K_bucket"]; split = d["split"]
    K = int(rank.max()) + 1
    print(f"[data] rows={len(X)} feats={X.shape[1]} K={K}  test_rows={int((split==1).sum())}")
    print(f"[seeds] {seeds}")
    print(f"[topk_list] {topk_list}")

    groups_all = group_by_mol(mol_id, K)
    g_test = [g for g in groups_all if split[g[0]] == 1]
    g_train = [g for g in groups_all if split[g[0]] == 0]
    print(f"[groups] train_mol={len(g_train)} test_mol={len(g_test)}  K={K}")

    rank_arr = rank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_mask = split == 0
    mu = X[tr_mask].mean(0); sd = X[tr_mask].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xt = torch.tensor(Xn, device=device)
    test_stack = torch.tensor(np.stack(g_test), device=device, dtype=torch.long)

    tr_groups_full = [g for g in g_train if correct[g].any()]
    rng = np.random.RandomState(12345)
    n_trg = len(tr_groups_full)
    val_n = int(0.15 * n_trg)
    perm0 = rng.permutation(n_trg)
    val_sel = perm0[:val_n]; tr_sel = perm0[val_n:]
    tr_stack = np.stack(tr_groups_full)
    tr_target = np.array([int(np.argmax(correct[g])) for g in tr_groups_full], dtype=np.int64)
    tr_idx_t = torch.tensor(tr_stack, device=device, dtype=torch.long)
    tr_tgt_t = torch.tensor(tr_target, device=device)
    val_idx_t = tr_idx_t[torch.tensor(val_sel, device=device)]
    val_tgt = tr_tgt_t[torch.tensor(val_sel, device=device)]
    tr2_idx_t = tr_idx_t[torch.tensor(tr_sel, device=device)]
    tr2_tgt_t = tr_tgt_t[torch.tensor(tr_sel, device=device)]
    M2 = tr2_idx_t.size(0)
    bs = 4096

    per_seed = []
    ens_score = None
    for s in seeds:
        state, vacc = train_one_seed(
            Xt, tr2_idx_t, tr2_tgt_t, val_idx_t, val_tgt,
            M2, bs, X.shape[1], args.h, args.lr, args.epochs, device, s)
        mdl = MLP(X.shape[1], h=args.h).to(device)
        mdl.load_state_dict(state); mdl.eval()
        with torch.no_grad():
            score = mdl(Xt[test_stack]).cpu().numpy()

        per_seed_metrics = topk_acc_per_bucket(
            g_test,
            scores_or_None=_expand_score_to_rows(score, g_test, len(X)),
            correct=correct, kbuckets=Kb, topk_list=topk_list, rank_tie_break=rank_arr)
        per_seed.append({"seed": s, "val_acc": vacc, "metrics": per_seed_metrics})
        print(f"  [seed {s}] val_acc={vacc:.4f}  test K_pos top-1/3/5 = "
              f"{per_seed_metrics['1']['K_pos']:.4f} / "
              f"{per_seed_metrics['3']['K_pos']:.4f} / "
              f"{per_seed_metrics['5']['K_pos']:.4f}")

        ens_score = score if ens_score is None else (ens_score + score)

    ens_metrics = topk_acc_per_bucket(
        g_test,
        scores_or_None=_expand_score_to_rows(ens_score, g_test, len(X)),
        correct=correct, kbuckets=Kb, topk_list=topk_list, rank_tie_break=rank_arr)
    print(f"  [ensemble n={len(seeds)}] test K_pos top-1/3/5 = "
          f"{ens_metrics['1']['K_pos']:.4f} / "
          f"{ens_metrics['3']['K_pos']:.4f} / "
          f"{ens_metrics['5']['K_pos']:.4f}")

    summary = {}
    for n in topk_list:
        vals = np.array([ps["metrics"][str(n)]["K_pos"] for ps in per_seed], dtype=np.float64)
        summary[str(n)] = {
            "K_pos_mean": float(vals.mean()),
            "K_pos_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "K_pos_per_seed": vals.tolist(),
        }
    print("\n[summary] mean ± s.d. (ddof=1) over seeds:")
    for n in topk_list:
        m = summary[str(n)]
        print(f"  top-{n}  K_pos = {m['K_pos_mean']*100:.2f} ± {m['K_pos_std']*100:.2f}  (per-seed {[f'{v*100:.2f}' for v in m['K_pos_per_seed']]})")

    out = {
        "n_test_mol": len(g_test),
        "K_candidates": K,
        "feats_path": args.feats,
        "seeds": seeds,
        "topk_list": topk_list,
        "hyperparams": {"epochs": args.epochs, "h": args.h, "lr": args.lr},
        "per_seed": per_seed,
        "ensemble": ens_metrics,
        "summary": summary,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
