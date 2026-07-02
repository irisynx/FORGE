"""产物感知重排 (§74) — 正式 decode mode 复用模块.

对 top-K class-flip 候选用 GT-free 信号重排选 top-1, 返回选中候选的 full-batch edit_pred.
两种打分:
  - handcraft: 产物合法性手工分 (plausibility_score), kept_mode mapped/mainpred.
  - learned (§74.route3): 19 维 GT-free 候选特征 → 轻量 MLP 打分 (LearnedReranker).

selection 只读预测产物 (smi_r + 预测 bond 变化生成), 不碰 GT.
特征计算 (candidate_features_mol) 与 dump_rerank_features.py 严格一致 (训练=推理同源).
"""
import numpy as np
import torch
import torch.nn as nn

FEAT_NAMES = [
    "ll", "ll_gap", "rank", "n_edit", "frac_break", "frac_form",
    "mean_conf", "min_conf", "conf_entropy",
    "plaus", "sanitizable", "n_radical", "n_frag",
    "mw_prod", "mw_gap_r", "ring_change", "heavy_gap_r",
    "rc_mean_on_edit", "rc_max_on_edit",
]
_CLS_BREAK = {2, 4, 6}
_CLS_FORM = {1, 3, 5}

def _frag_smiles(canon, main_only):
    if not canon:
        return canon
    if not main_only or "." not in canon:
        return canon
    return max(canon.split("."), key=len)

def plausibility_score(canon, main_only=False):
    if not canon:
        return -10.0
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    smi = _frag_smiles(canon, main_only)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return -8.0
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return -5.0
    radicals = sum(a.GetNumRadicalElectrons() for a in m.GetAtoms())
    try:
        n_frag = len(Chem.GetMolFrags(m))
    except Exception:
        n_frag = 1
    return -2.0 * radicals - 0.5 * max(0, n_frag - 1)

def _rdkit_props(smi):
    """(MW, n_ring, n_heavy, n_radical, n_frag, sanitizable); 失败 None."""
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
    return (mw, m.GetRingInfo().NumRings(), m.GetNumHeavyAtoms(),
            sum(a.GetNumRadicalElectrons() for a in m.GetAtoms()),
            (len(Chem.GetMolFrags(m)) if m.GetNumAtoms() else 1), san)

def candidate_features_mol(smi_r, cand_local, ep_list_mol, log_probs_mol, probs_mol,
                           rc_loc, n_loc, canon_fn, cls2delta):
    """返回 [K, 19] np.float32, 与 dump_rerank_features.py 内层逐行严格一致.
       ep_list_mol: list 长 K, 每个 = 该 mol 的 cls 数组 [k]; log_probs_mol/probs_mol: [k,7]."""
    K = len(ep_list_mol)
    k = log_probs_mol.shape[0]
    rprops = _rdkit_props(smi_r)
    r_mw = rprops[0] if rprops else 0.0
    r_ring = rprops[1] if rprops else 0
    r_heavy = rprops[2] if rprops else 0
    feats = np.zeros((K, 19), dtype=np.float32)
    ll_r1 = None
    for r in range(K):
        cls = ep_list_mol[r]
        ll = float(log_probs_mol[np.arange(k), cls].sum())
        if r == 0:
            ll_r1 = ll
        nz = cls != 0
        n_edit = int(nz.sum())
        if n_edit > 0:
            conf = probs_mol[np.arange(k), cls][nz]
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
        changes = [(int(cand_local[e, 0]), int(cand_local[e, 1]), cls2delta.get(int(cls[e]), 0))
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
        feats[r] = [
            ll, ll - (ll_r1 if ll_r1 is not None else ll), float(r),
            float(n_edit), frac_break, frac_form,
            mean_conf, min_conf, ent,
            plaus, float(san), float(nrad), float(nfrag),
            mw_p, mw_p - r_mw, float(ring_p - r_ring), float(heavy_p - r_heavy),
            rc_mean, rc_max,
        ]
    return feats

class _MLP(nn.Module):
    def __init__(self, F, h):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(F, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

class LearnedReranker:
    """加载 reranker_train.py 存的 .pt (state_dict + mu + sd + feat_names + h)."""

    def __init__(self, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.feat_names = list(ckpt["feat_names"])
        assert self.feat_names == FEAT_NAMES, "feat_names 不一致, 训练/推理特征 drift!"
        self.mu = np.asarray(ckpt["mu"], np.float32)
        self.sd = np.asarray(ckpt["sd"], np.float32)
        self.model = _MLP(len(self.feat_names), ckpt["h"]).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device = device

    def score(self, feats):
        """feats [K,F] np -> [K] np 分数 (越大越优)."""
        xn = (feats - self.mu) / self.sd
        with torch.no_grad():
            s = self.model(torch.tensor(xn, dtype=torch.float32, device=self.device))
        return s.cpu().numpy()

def build_reranked_edit_pred(outputs, batch, T, edit_pred_list, kept_mode="mapped",
                             learned_model=None):
    """返回 (reranked_edit_pred [full-batch LongTensor], n_changed).
       learned_model 给定 → 用 LearnedReranker 打分; 否则用 handcraft plausibility (kept_mode)."""
    device = outputs["edit_logits"].device
    K = len(edit_pred_list)
    cand_edges_list = outputs.get("cand_edges", [])
    edge_counts = [int(e.size(0)) for e in cand_edges_list]
    batch_size = len(edge_counts)

    ep_cpu = [ep.detach().cpu().numpy() for ep in edit_pred_list]
    selected_t = edit_pred_list[0].detach().cpu().clone()
    smi_r_list = getattr(batch, "smi_r_list", [""] * batch_size)
    cls2delta = T._CLS_TO_DELTA_INT
    canon_fn = T._v26_product_canonical
    main_only = (kept_mode == "mainpred")

    if learned_model is not None:
        edit_logits = outputs["edit_logits"].float()
        log_probs = torch.log_softmax(edit_logits, dim=-1).cpu().numpy()
        probs = torch.softmax(edit_logits, dim=-1).cpu().numpy()
        rc_probs = outputs["rc_probs"].float().cpu().numpy()
        batch_idx = batch.batch.cpu().numpy()
        node_counts = np.bincount(batch_idx, minlength=batch_size)
        node_offsets = np.concatenate([[0], np.cumsum(node_counts)[:-1]])

    n_changed = 0
    offset = 0
    for mi in range(batch_size):
        k = edge_counts[mi] if mi < len(edge_counts) else 0
        if k == 0:
            continue
        smi_r = smi_r_list[mi] if mi < len(smi_r_list) else ""
        cand_local = cand_edges_list[mi].cpu().numpy()
        if learned_model is not None:
            n_loc = int(node_counts[mi]); start_loc = int(node_offsets[mi])
            rc_loc = rc_probs[start_loc:start_loc + n_loc]
            ep_list_mol = [ep_cpu[r][offset:offset + k] for r in range(K)]
            feats = candidate_features_mol(
                smi_r, cand_local, ep_list_mol,
                log_probs[offset:offset + k], probs[offset:offset + k],
                rc_loc, n_loc, canon_fn, cls2delta)
            scores = learned_model.score(feats)
            best_r = int(max(range(K), key=lambda r: (scores[r], -r)))
        else:
            best_r, best_s = 0, None
            for r in range(K):
                cls = ep_cpu[r][offset:offset + k]
                changes = [(int(cand_local[e, 0]), int(cand_local[e, 1]), cls2delta.get(int(cls[e]), 0))
                           for e in range(k) if cls[e] != 0]
                try:
                    canon = canon_fn(smi_r, changes, None)
                except Exception:
                    canon = ""
                s = plausibility_score(canon, main_only=main_only)
                if best_s is None or s > best_s:
                    best_s, best_r = s, r
        if best_r != 0:
            selected_t[offset:offset + k] = torch.from_numpy(ep_cpu[best_r][offset:offset + k])
            n_changed += 1
        offset += k

    return selected_t.to(device), n_changed
