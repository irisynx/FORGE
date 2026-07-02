"""Fig4 (a) Panel — Compute top-1/3/5 for the trained top-15 reranker MLP.

Loads a `reranker_mlp.pt` checkpoint + `feats.npz` produced by
`dump_rerank_features.py` + `train_eval_reranker.py` and computes
top-K K_pos for:
  - greedy   (base model rank ascending)
  - handcraft (plaus desc, tie-break by rank asc)
  - learned  (trained MLP score desc)
  - oracle   (a top-K of oracle is degenerate: it's the same for all K ≥ 1
              — we still log it for sanity)

Per-bucket and overall (K=0..5) accuracies are written to JSON.

Usage:
    .venv/bin/python tools/train_eval_reranker.py \\
        --feats analysis_reports/rerank_feats_v3_top15/feats.npz \\
        --ckpt  analysis_reports/rerank_feats_v3_top15/reranker_mlp.pt \\
        --out   analysis_reports/v3_rerank15_topk_F4/topk_table.json \\
        --n_seed 3 --topk 1,3,5,10
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

BUCKETS = [0, 2, 3, 4, 5]

class MLP(nn.Module):
    def __init__(self, F, h=64, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

def group_by_mol(mol_id, K):
    """Return list of (row indices) per mol; keep mols with exactly K candidates."""
    order = np.argsort(mol_id, kind="stable")
    groups = []
    i = 0
    n = len(mol_id)
    while i < n:
        j = i
        while j < n and mol_id[order[j]] == mol_id[order[i]]:
            j += 1
        idx = order[i:j]
        if len(idx) == K:
            groups.append(idx)
        i = j
    return groups

def topk_acc_per_bucket(groups, scores_or_None, correct, kbuckets, topk_list,
                        rank_tie_break=None, fallback_rank=None):
    """For each `n` in topk_list, compute per-bucket + overall (incl K=0) +
    K_pos (excluding K=0).

    scores_or_None: per-row score (higher=better). If None, use fallback_rank (smaller=better).
    rank_tie_break: per-row rank (smaller=better), tie-break for handcraft.
    """
    out = {}
    for n in topk_list:
        agg = {b: [0, 0] for b in BUCKETS}
        total_hit = 0
        total_cnt = 0
        for g in groups:
            if scores_or_None is not None:
                s = scores_or_None[g]
                if rank_tie_break is not None:
                    rt = rank_tie_break[g]
                    order = np.lexsort((rt, -s))
                else:
                    order = np.argsort(-s, kind="stable")
            else:

                order = np.argsort(fallback_rank[g], kind="stable")
            sel = order[:n]
            hit = int(correct[g][sel].any())
            b = int(kbuckets[g[0]])
            agg[b][0] += hit
            agg[b][1] += 1
            total_hit += hit
            total_cnt += 1

        kpos_c = sum(agg[b][0] for b in BUCKETS if b != 0)
        kpos_n = sum(agg[b][1] for b in BUCKETS if b != 0)
        out[str(n)] = {
            "K_pos": kpos_c / kpos_n if kpos_n else 0.0,
            "overall_incl_K0": total_hit / total_cnt if total_cnt else 0.0,
            "per_bucket": {str(b): (agg[b][0] / agg[b][1] if agg[b][1] else 0.0) for b in BUCKETS},
        }
    return out

def train_one_seed(Xt, tr2_idx_t, tr2_tgt_t, val_idx_t, val_tgt,
                   M2, bs, F, h, lr, epochs, device, seed):
    torch.manual_seed(seed)
    model = MLP(F, h=h).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    best_val = -1.0
    best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(M2, device=device)
        for b in range(0, M2, bs):
            bi = perm[b:b + bs]
            s = model(Xt[tr2_idx_t[bi]])
            loss = ce(s, tr2_tgt_t[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vlogit = model(Xt[val_idx_t])
            vacc = float((vlogit.argmax(1) == val_tgt).float().mean())
        if vacc > best_val:
            best_val = vacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="feats.npz from dump_rerank_features.py")
    ap.add_argument("--ckpt", default="", help="(optional) reranker_mlp.pt; if absent will retrain")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--topk", default="1,3,5,10", help="comma list of N for top-N evaluation")
    ap.add_argument("--n_seed", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--h", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    topk_list = [int(x) for x in args.topk.split(",") if x.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    d = np.load(args.feats, allow_pickle=True)
    X = d["X"].astype(np.float32)
    fn = list(d["feat_names"])
    mol_id = d["mol_id"]; rank = d["rank"]; correct = d["correct"].astype(np.int64)
    Kb = d["K_bucket"]; split = d["split"]
    K = int(rank.max()) + 1
    print(f"[data] rows={len(X)} feats={X.shape[1]} K={K}  test_rows={int((split==1).sum())}")
    print(f"[topk_list] {topk_list}")

    groups_all = group_by_mol(mol_id, K)
    g_test = [g for g in groups_all if split[g[0]] == 0 or split[g[0]] == 1]
    g_test = [g for g in groups_all if split[g[0]] == 1]
    g_train = [g for g in groups_all if split[g[0]] == 0]
    print(f"[groups] train_mol={len(g_train)} test_mol={len(g_test)}  K={K}")

    plaus_col = fn.index("plaus")
    rank_arr = rank

    print(f"[baseline] greedy / handcraft / oracle top-{topk_list} on test ({len(g_test)} mols)")
    res = {
        "n_test_mol": len(g_test),
        "K_candidates": K,
        "feats_path": args.feats,
        "ckpt_path": args.ckpt,
        "topk_list": topk_list,
        "n_seed": args.n_seed,
        "methods": {},
    }

    res["methods"]["greedy"] = topk_acc_per_bucket(
        g_test, scores_or_None=None, correct=correct, kbuckets=Kb,
        topk_list=topk_list, fallback_rank=rank_arr)
    print(f"  greedy  K_pos top-1/3/5 = "
          f"{res['methods']['greedy']['1']['K_pos']:.4f} / "
          f"{res['methods']['greedy']['3']['K_pos']:.4f} / "
          f"{res['methods']['greedy']['5']['K_pos']:.4f}")

    res["methods"]["handcraft"] = topk_acc_per_bucket(
        g_test, scores_or_None=X[:, plaus_col], correct=correct, kbuckets=Kb,
        topk_list=topk_list, rank_tie_break=rank_arr)
    print(f"  handc.  K_pos top-1/3/5 = "
          f"{res['methods']['handcraft']['1']['K_pos']:.4f} / "
          f"{res['methods']['handcraft']['3']['K_pos']:.4f} / "
          f"{res['methods']['handcraft']['5']['K_pos']:.4f}")

    res["methods"]["oracle"] = topk_acc_per_bucket(
        g_test, scores_or_None=correct.astype(np.float32), correct=correct, kbuckets=Kb,
        topk_list=topk_list, rank_tie_break=rank_arr)
    print(f"  oracle  K_pos top-1/3/5 = "
          f"{res['methods']['oracle']['1']['K_pos']:.4f} / "
          f"{res['methods']['oracle']['3']['K_pos']:.4f} / "
          f"{res['methods']['oracle']['5']['K_pos']:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_mask = split == 0
    mu = X[tr_mask].mean(0); sd = X[tr_mask].std(0) + 1e-6
    Xn = (X - mu) / sd
    Xt = torch.tensor(Xn, device=device)

    test_stack = torch.tensor(np.stack(g_test), device=device, dtype=torch.long)

    use_ckpt = bool(args.ckpt) and os.path.exists(args.ckpt)
    ens_score = None

    if use_ckpt:
        print(f"[ckpt] load {args.ckpt}")
        ck = torch.load(args.ckpt, map_location=device)

        if "state_dict" in ck:
            state = ck["state_dict"]
            mu_ck = ck.get("mu", mu); sd_ck = ck.get("sd", sd)
            h_ck = ck.get("h", args.h)
        else:
            state = ck
            mu_ck, sd_ck, h_ck = mu, sd, args.h

        Xn_eval = (X - mu_ck) / (sd_ck + 1e-9)
        Xt_eval = torch.tensor(Xn_eval.astype(np.float32), device=device)

        model = MLP(X.shape[1], h=h_ck).to(device)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            ens_score = model(Xt_eval[test_stack]).cpu().numpy()
        print("[ckpt] single seed (the saved best); n_seed override -> 1")
        n_seed_used = 1
    else:

        print("[no ckpt] retraining MLP from scratch")
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
        for s in range(args.n_seed):
            state, vacc = train_one_seed(
                Xt, tr2_idx_t, tr2_tgt_t, val_idx_t, val_tgt,
                M2, bs, X.shape[1], args.h, args.lr, args.epochs, device, args.seed + s)
            mdl = MLP(X.shape[1], h=args.h).to(device)
            mdl.load_state_dict(state); mdl.eval()
            with torch.no_grad():
                score = mdl(Xt[test_stack]).cpu().numpy()
            ens_score = score if ens_score is None else (ens_score + score)
            print(f"  [seed {args.seed+s}] best val acc = {vacc:.4f}")
        n_seed_used = args.n_seed

    res["methods"]["learned"] = topk_acc_per_bucket(
        g_test,
        scores_or_None=_expand_score_to_rows(ens_score, g_test, len(X)),
        correct=correct, kbuckets=Kb, topk_list=topk_list, rank_tie_break=rank_arr)
    print(f"  learned K_pos top-1/3/5 = "
          f"{res['methods']['learned']['1']['K_pos']:.4f} / "
          f"{res['methods']['learned']['3']['K_pos']:.4f} / "
          f"{res['methods']['learned']['5']['K_pos']:.4f}")

    res["n_seed_used"] = n_seed_used
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {args.out}")

def _expand_score_to_rows(score_MK, g_test, n_total_rows):
    """Score is shape [Mt, K]. Expand to a length-n_total_rows array indexed
    by raw feats.npz row id so the existing topk_acc helper can use it."""
    out = np.full(n_total_rows, -np.inf, dtype=np.float32)
    for i, g in enumerate(g_test):
        out[g] = score_MK[i]
    return out

if __name__ == "__main__":
    main()
