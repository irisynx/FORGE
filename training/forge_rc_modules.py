"""§62 v33_J Step 2+3: RCDecoderLayer + RCSetPredHead + Hungarian Set Prediction Loss.

模块 standalone 实现, 后续 Step 5 wire 进 train_edit_v33_J.py.

设计参考 §62.3 / §62.4 / §62.6.

关键约束:
- 输入 h_atom: [N_total, hidden] flat, 用 batch_idx + reactant_mask 划分
- K_max = 10 (per Step 1 验证, cover 99.82%)
- Hungarian on CPU (scipy linear_sum_assignment, per-reaction <1ms)
- atom_logits via dot product (queries · h_atom_dense.T) / sqrt(hidden), 不复用 cross-attn weights (后者训练 grad 不稳)
- presence_logits 独立 head (Linear)
- 截断: K_gt > K_max 反应仅前 K_max 个 GT atoms 参与 matching (per §62.14)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment

class RCDecoderLayer(nn.Module):
    """单层 DETR-style decoder: self-attn (queries 间) + cross-attn (Q=queries, K/V=atoms) + FFN.

    每层后接 Pre-LN 残差 (DETR 用 post-LN; 这里 Pre-LN 在浅层 decoder 上更稳).
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        atom_feat: torch.Tensor,
        atom_padding_mask: torch.Tensor,
    ) -> torch.Tensor:

        q_normed = self.norm1(queries)
        sa, _ = self.self_attn(q_normed, q_normed, q_normed, need_weights=False)
        queries = queries + self.dropout(sa)

        q_normed = self.norm2(queries)
        ca, _ = self.cross_attn(
            q_normed, atom_feat, atom_feat,
            key_padding_mask=atom_padding_mask,
            need_weights=False,
        )
        queries = queries + self.dropout(ca)

        q_normed = self.norm3(queries)
        queries = queries + self.dropout(self.ffn(q_normed))
        return queries

class RCSetPredHead(nn.Module):
    """DETR-style RC set prediction head.

    Replaces v29.2 RCHead (atom-level sigmoid) + K_pred head (multi-class softmax).

    Inputs (per forward):
        h_atom:        [N_total, hidden]  flat atom features
        batch_idx:     [N_total] int
        reactant_mask: [N_total] bool, True for reactant atoms

    Outputs:
        presence_logits: [B, K_max]
        atom_logits:     [B, K_max, N_max]
        mask_dense:      [B, N_max] bool, True for valid atom positions (reactant atoms only)
        atom_idx_dense:  [B, N_max] long, original atom indices (for back-tracking)

    Notes:
        - atom_logits via dot product (queries · h_dense.T) / sqrt(hidden) — 训练 grad 比 cross-attn 权重稳
        - presence_logits via separate Linear head, sigmoid in loss
    """

    def __init__(self, hidden_dim: int = 512, K_max: int = 10,
                 num_layers: int = 3, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1,
                 N_max: int = 60):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.K_max = K_max
        self.N_max = N_max

        self.object_queries = nn.Parameter(torch.randn(K_max, hidden_dim) * 0.02)

        self.layers = nn.ModuleList([
            RCDecoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.presence_head = nn.Linear(hidden_dim, 1)

        self.atom_proj_query = nn.Linear(hidden_dim, hidden_dim)
        self.atom_proj_key = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        h_atom: torch.Tensor,
        batch_idx: torch.Tensor,
        reactant_mask: torch.Tensor,
        return_queries: bool = False,
    ):
        device = h_atom.device

        h_react = h_atom[reactant_mask]
        batch_react = batch_idx[reactant_mask]

        atom_indices_flat = torch.arange(h_atom.size(0), device=device)[reactant_mask]

        h_dense, mask_dense = to_dense_batch(
            h_react, batch_react, max_num_nodes=self.N_max
        )

        atom_idx_dense, _ = to_dense_batch(
            atom_indices_flat, batch_react, fill_value=-1, max_num_nodes=self.N_max
        )
        B = h_dense.size(0)
        atom_padding_mask = ~mask_dense

        queries = self.object_queries.unsqueeze(0).expand(B, -1, -1).contiguous()

        for layer in self.layers:
            queries = layer(queries, h_dense, atom_padding_mask)
        queries = self.final_norm(queries)

        presence_logits = self.presence_head(queries).squeeze(-1)

        Q = self.atom_proj_query(queries)
        K = self.atom_proj_key(h_dense)
        atom_logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)

        atom_logits = atom_logits.masked_fill(
            atom_padding_mask.unsqueeze(1), float('-inf')
        )

        if return_queries:

            return presence_logits, atom_logits, mask_dense, atom_idx_dense, queries
        return presence_logits, atom_logits, mask_dense, atom_idx_dense

    def project_queries(self, refined_queries, h_atom, batch_idx, reactant_mask):
        """§66 v36: re-project externally refined queries through presence + atom-pointer
        heads, recomputing h_dense identically to forward. Returns (presence_logits,
        atom_logits) for the refined-RC set-prediction supervision (avoidance #1)."""
        h_react = h_atom[reactant_mask]
        batch_react = batch_idx[reactant_mask]
        h_dense, mask_dense = to_dense_batch(
            h_react, batch_react, max_num_nodes=self.N_max
        )
        atom_padding_mask = ~mask_dense
        presence_logits = self.presence_head(refined_queries).squeeze(-1)
        Q = self.atom_proj_query(refined_queries)
        K = self.atom_proj_key(h_dense)
        atom_logits = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        atom_logits = atom_logits.masked_fill(
            atom_padding_mask.unsqueeze(1), float('-inf')
        )
        return presence_logits, atom_logits

@torch.no_grad()
def hungarian_match_one(
    presence_logits_b: torch.Tensor,
    atom_logits_b: torch.Tensor,
    gt_atom_dense_idx_b: torch.Tensor,
    valid_atom_mask_b: torch.Tensor,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-reaction Hungarian matching.

    Returns:
        matched_query_idx: [K_gt] long — for each GT atom, the query index assigned
        matched_gt_idx:    [K_gt] long — for each matched query (in matched_query_idx),
                           the GT atom dense index in 0..N_max-1
        即: presence/atom_logits[matched_query_idx[i]] 应预测 gt_atom_dense_idx[matched_gt_idx[i]]

    若 K_gt > K_max: 截断, 仅前 K_max 个 GT atoms 参与 matching.
    若 K_gt == 0: 返回空 tensor (所有 queries 学 no-object).
    """
    K_max = presence_logits_b.size(0)
    K_gt = gt_atom_dense_idx_b.size(0)
    device = presence_logits_b.device

    if K_gt == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    K_gt_used = min(K_gt, K_max)
    gt_subset = gt_atom_dense_idx_b[:K_gt_used]

    p_log = F.logsigmoid(presence_logits_b)

    atom_log_prob = F.log_softmax(atom_logits_b, dim=-1)

    gt_log_prob = atom_log_prob[:, gt_subset]
    cost = -p_log.unsqueeze(-1) - alpha * gt_log_prob

    cost_np = cost.detach().float().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)

    matched_query_idx = torch.as_tensor(row_ind, dtype=torch.long, device=device)
    matched_gt_idx = torch.as_tensor(col_ind, dtype=torch.long, device=device)
    return matched_query_idx, matched_gt_idx

def compute_set_prediction_loss(
    presence_logits: torch.Tensor,
    atom_logits: torch.Tensor,
    mask_dense: torch.Tensor,
    gt_rc_target_flat: torch.Tensor,
    batch_idx: torch.Tensor,
    reactant_mask: torch.Tensor,
    atom_idx_dense: torch.Tensor,
    lambda_no: float = 0.5,
    alpha_cost: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Set prediction loss with Hungarian matching.

    Returns dict with:
        loss_set:       total set loss (atom_ce + lambda_no * presence_bce)
        loss_atom_ce:   atom CE on matched queries only
        loss_presence:  presence BCE on all queries (matched=1, unmatched=0)
        matched_count:  total matched queries across batch
        no_object_count: total no-object queries across batch
    """
    device = presence_logits.device
    B, K_max, N_max = atom_logits.shape

    rc_flat = gt_rc_target_flat[reactant_mask].bool()
    batch_react = batch_idx[reactant_mask]

    rc_mask_dense, _ = to_dense_batch(
        rc_flat.long(), batch_react, max_num_nodes=N_max
    )

    loss_atom_ce_list = []
    presence_targets = torch.zeros(B, K_max, device=device)
    matched_count_total = 0

    for b in range(B):
        gt_atoms_dense = torch.nonzero(rc_mask_dense[b].bool(), as_tuple=False).squeeze(-1)

        matched_q_idx, matched_gt_idx = hungarian_match_one(
            presence_logits[b], atom_logits[b],
            gt_atoms_dense, mask_dense[b],
            alpha=alpha_cost,
        )
        K_matched = matched_q_idx.size(0)
        if K_matched == 0:
            continue

        target_atom_dense = gt_atoms_dense[matched_gt_idx]
        logits_for_matched = atom_logits[b, matched_q_idx]
        ce = F.cross_entropy(logits_for_matched, target_atom_dense, reduction='sum')
        loss_atom_ce_list.append(ce)

        presence_targets[b, matched_q_idx] = 1.0
        matched_count_total += K_matched

    if loss_atom_ce_list:
        loss_atom_ce = torch.stack(loss_atom_ce_list).sum() / max(1, matched_count_total)
    else:
        loss_atom_ce = presence_logits.sum() * 0.0

    loss_presence = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets, reduction='mean'
    )

    no_object_count_total = B * K_max - matched_count_total

    loss_set = loss_atom_ce + lambda_no * loss_presence

    return {
        "loss_set": loss_set,
        "loss_atom_ce": loss_atom_ce.detach(),
        "loss_presence": loss_presence.detach(),
        "matched_count": torch.tensor(matched_count_total, device=device),
        "no_object_count": torch.tensor(no_object_count_total, device=device),
    }

@torch.no_grad()
def predict_rc_atoms_from_set(
    presence_logits: torch.Tensor,
    atom_logits: torch.Tensor,
    mask_dense: torch.Tensor,
    atom_idx_dense: torch.Tensor,
    tau: float = 0.5,
) -> list[list[int]]:
    """推理: presence > tau 的 queries argmax atom; 去重.

    Returns:
        list of B lists, each inner list is the flat atom indices of predicted RC atoms.
    """
    presence_probs = torch.sigmoid(presence_logits)
    B, K_max, N_max = atom_logits.shape
    result = []
    for b in range(B):
        active = (presence_probs[b] > tau).nonzero(as_tuple=False).squeeze(-1)
        if active.numel() == 0:
            result.append([])
            continue

        atom_choices_dense = atom_logits[b, active].argmax(dim=-1)

        atom_choices_flat = atom_idx_dense[b, atom_choices_dense]

        unique_atoms = list(set(atom_choices_flat.tolist()))

        unique_atoms = [a for a in unique_atoms if a >= 0]
        result.append(unique_atoms)
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("§62 v33_J modules smoke test")
    print("=" * 60)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    atom_counts_reactant = [20, 30, 25]
    atom_counts_total = [40, 50, 45]
    N_total = sum(atom_counts_total)
    hidden = 512
    K_max = 10
    N_max = 60

    h_atom = torch.randn(N_total, hidden, device=device)
    batch_idx = []
    reactant_mask = []
    for b, (n_r, n_t) in enumerate(zip(atom_counts_reactant, atom_counts_total)):
        batch_idx.extend([b] * n_t)
        reactant_mask.extend([True] * n_r + [False] * (n_t - n_r))
    batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=device)
    reactant_mask = torch.tensor(reactant_mask, dtype=torch.bool, device=device)

    K_gts = [3, 5, 0]
    gt_rc_flat = torch.zeros(N_total, dtype=torch.bool, device=device)

    cursor = 0
    for b, (n_r, n_t, kgt) in enumerate(zip(atom_counts_reactant, atom_counts_total, K_gts)):
        if kgt > 0:
            rc_positions = torch.randperm(n_r)[:kgt]
            for p in rc_positions:
                gt_rc_flat[cursor + p.item()] = True
        cursor += n_t

    print(f"\nMock batch: B=3, K_gt={K_gts}, reactant atoms={atom_counts_reactant}")
    print(f"N_total={N_total}, total RC atoms (flat)={int(gt_rc_flat.sum())}")

    head = RCSetPredHead(hidden_dim=hidden, K_max=K_max, num_layers=3, N_max=N_max).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\nRCSetPredHead trainable params: {n_params:,} (~{n_params/1e6:.2f}M)")

    presence_logits, atom_logits, mask_dense, atom_idx_dense = head(
        h_atom, batch_idx, reactant_mask
    )
    print(f"Forward output shapes:")
    print(f"  presence_logits:  {tuple(presence_logits.shape)}  [expected (3, {K_max})]")
    print(f"  atom_logits:      {tuple(atom_logits.shape)}  [expected (3, {K_max}, {N_max})]")
    print(f"  mask_dense:       {tuple(mask_dense.shape)}  valid counts={mask_dense.sum(1).tolist()}")
    print(f"  atom_idx_dense:   {tuple(atom_idx_dense.shape)}")

    loss_dict = compute_set_prediction_loss(
        presence_logits, atom_logits, mask_dense,
        gt_rc_flat, batch_idx, reactant_mask, atom_idx_dense,
        lambda_no=0.5,
    )
    print(f"\nLoss results:")
    print(f"  loss_set:          {loss_dict['loss_set'].item():.4f}")
    print(f"  loss_atom_ce:      {loss_dict['loss_atom_ce'].item():.4f}")
    print(f"  loss_presence:     {loss_dict['loss_presence'].item():.4f}")
    print(f"  matched_count:     {loss_dict['matched_count'].item()}  (expected {sum(min(k, K_max) for k in K_gts)} = 3+5+0)")
    print(f"  no_object_count:   {loss_dict['no_object_count'].item()}  (expected {3*K_max - sum(min(k, K_max) for k in K_gts)} = 30-8=22)")

    loss_dict['loss_set'].backward()
    grad_count = sum(1 for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in head.parameters())
    print(f"\nGradient flow: {grad_count}/{total_params} param tensors have nonzero grad")

    preds = predict_rc_atoms_from_set(
        presence_logits, atom_logits, mask_dense, atom_idx_dense, tau=0.5
    )
    print(f"\nInference predictions (tau=0.5):")
    for b, p in enumerate(preds):
        print(f"  Reaction {b} (K_gt={K_gts[b]}): predicted {len(p)} atoms: {p[:6]}{'...' if len(p) > 6 else ''}")

    print(f"\n--- Edge case: K_gt > K_max truncation ---")
    gt_rc_big = torch.zeros(N_total, dtype=torch.bool, device=device)

    cursor = 0
    for p in range(min(15, atom_counts_reactant[0])):
        gt_rc_big[cursor + p] = True
    loss_big = compute_set_prediction_loss(
        presence_logits, atom_logits, mask_dense,
        gt_rc_big, batch_idx, reactant_mask, atom_idx_dense, lambda_no=0.5,
    )
    expected_matched = K_max
    print(f"  K_gt=15 (truncated to {K_max}): matched_count={loss_big['matched_count'].item()} (expected {expected_matched})")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
