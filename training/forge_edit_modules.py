"""§64 v34_E1 Step 2: DETREditHead + DETREditDecoderLayer.

镜像 v33_J RC 端 DETR set prediction 到 Edit 端: 每条 cand edge 一个 object query,
cross-attn 到 RC anchor pool, 输出 7-class delta logit.

关键约束 (per §64.4):
- Pre-LN Transformer decoder (norm_first=True, 与 v33_J RCDecoderLayer 一致)
- Self-attn within same mol (per-mol forward 切片实现 mol-aware)
- Cross-attn: edge_query → mol's RC anchor pool (来自 v33_J DETR RC head post-decoder)
- 末层 nn.Linear(hidden, 7), warmstart 用 v29.2 EditClassifier 末层权重 init
  → fresh head 起步等价 v29.2 sigmoid baseline (兑现 feedback_warmstart_fresh_module_lr)
- K=0 反应天然 short-circuit (cand edges 为空, mol_mask 全 False)
- 单 cand edge 反应 → 1-query DETR, FFN/cross-attn 仍正常 (兑现 §64.4 corner case)
- rc_anchor_mask 全 padding 防御: skip 避免 cross-attn NaN

不引入 local prior / SMARTS / atom-env (per feedback_local_prior_gnn_redundancy).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class DETREditDecoderLayer(nn.Module):
    """单层 Edit-side DETR decoder.

    forward(edge_q, rc_anchor, anchor_padding_mask):
        - Self-attn over edges within same mol (per-mol slice 已隔离, 不需 mol mask)
        - Cross-attn: Q=edges, K/V=rc_anchor pool, key_padding_mask=anchor_padding
        - FFN (Linear → GELU → Dropout → Linear)
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
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
        edge_q: torch.Tensor,
        rc_anchor: torch.Tensor,
        anchor_padding_mask: torch.Tensor,
        edge_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        x = self.norm1(edge_q)
        sa, _ = self.self_attn(
            x, x, x,
            key_padding_mask=edge_padding_mask,
            need_weights=False,
        )
        edge_q = edge_q + self.dropout(sa)

        x = self.norm2(edge_q)
        ca, _ = self.cross_attn(
            x, rc_anchor, rc_anchor,
            key_padding_mask=anchor_padding_mask,
            need_weights=False,
        )
        edge_q = edge_q + self.dropout(ca)

        x = self.norm3(edge_q)
        edge_q = edge_q + self.dropout(self.ffn(x))
        return edge_q

class DETREditHead(nn.Module):
    """Per-cand-edge object query DETR head, replaces v29.2 EditClassifier.

    Inputs (per forward):
        h_edge:         [K_total_cand, hidden]  EdgeTransformer 输出 (cand edge embeddings)
        edge_mol_ids:   [K_total_cand] long     cand edge → mol idx mapping (0..B-1)
        rc_anchor_emb:  [B, K_max_rc, hidden]   v33_J DETR RC head post-decoder embeddings
        rc_anchor_mask: [B, K_max_rc] bool      True = padding (与 PyTorch attn mask 约定一致)

    Output:
        edit_logits:    [K_total_cand, n_classes=7]

    Warmstart:
        init_edit_classifier_from_v29(weight, bias) 把 v29.2 EditClassifier 末层权重复制过来,
        让 fresh head 起步等价 sigmoid baseline (per §64.5 + feedback_warmstart_fresh_module_lr).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_decoder_layers: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        n_classes: int = 7,
        cls_bottleneck: int = 256,
        cls_dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList([
            DETREditDecoderLayer(hidden_dim, n_heads, ffn_dim, dropout)
            for _ in range(n_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        self.edit_classifier = nn.Sequential(
            nn.Linear(hidden_dim, cls_bottleneck),
            nn.LayerNorm(cls_bottleneck),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_bottleneck, n_classes),
        )

        with torch.no_grad():
            self.edit_classifier[-1].bias[0] = 1.5
            self.edit_classifier[-1].bias[1:] = -0.25

    def init_edit_classifier_from_v29(self, v29_mlp_state_dict: dict) -> int:
        """Copy v29.2 EditClassifier.mlp 子 state_dict → detr_edit_head.edit_classifier.

        v29.2 EditClassifier 结构 (train_edit_v33_J.py:4002):
            mlp[0]: Linear(hidden, 256)        — keys: mlp.0.weight, mlp.0.bias
            mlp[1]: LayerNorm(256)              — keys: mlp.1.weight, mlp.1.bias
            mlp[2]: GELU (stateless)
            mlp[3]: Dropout (stateless)
            mlp[4]: Linear(256, n_classes)      — keys: mlp.4.weight, mlp.4.bias

        v34_E1 DETREditHead.edit_classifier 用 nn.Sequential 镜像同结构:
            edit_classifier[0]: Linear → keys: 0.weight, 0.bias
            edit_classifier[1]: LayerNorm → keys: 1.weight, 1.bias
            edit_classifier[4]: Linear → keys: 4.weight, 4.bias

        v29_mlp_state_dict: 形如 {'mlp.0.weight': T, 'mlp.0.bias': T, ...} (来自 v33_J ckpt
        中 edit_classifier.mlp.* 子集).

        Returns: copied tensor 数 (期望 6: 2 Linear × 2 + 1 LayerNorm × 2 = 6).
        """
        copied = 0
        key_remap = {
            "mlp.0.weight": "0.weight", "mlp.0.bias": "0.bias",
            "mlp.1.weight": "1.weight", "mlp.1.bias": "1.bias",
            "mlp.4.weight": "4.weight", "mlp.4.bias": "4.bias",
        }
        target_sd = dict(self.edit_classifier.state_dict())
        for src_k, dst_k in key_remap.items():
            if src_k not in v29_mlp_state_dict:
                continue
            v = v29_mlp_state_dict[src_k]
            if dst_k not in target_sd:
                continue
            if v.shape != target_sd[dst_k].shape:
                raise ValueError(
                    f"shape mismatch on {src_k}: v29 {tuple(v.shape)} vs "
                    f"detr {tuple(target_sd[dst_k].shape)}"
                )
            target_sd[dst_k] = v.clone()
            copied += 1
        self.edit_classifier.load_state_dict(target_sd)
        return copied

    def forward(
        self,
        h_edge: torch.Tensor,
        edge_mol_ids: torch.Tensor,
        rc_anchor_emb: torch.Tensor,
        rc_anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """批量化版本 (2026-05-27 perf fix):
        用 to_dense_batch 把 cand edges pack 到 [B, K_max_cand, D] 单 kernel forward,
        消除 per-mol Python loop (旧版 1.05 s/step 主因之一).
        """
        device = h_edge.device
        B = rc_anchor_emb.size(0)
        K_total = h_edge.size(0)

        edit_logits = torch.zeros(
            K_total, self.n_classes, device=device, dtype=h_edge.dtype
        )

        if K_total == 0:
            return edit_logits

        h_dense, cand_mask = to_dense_batch(h_edge, edge_mol_ids, batch_size=B)

        has_cand = cand_mask.any(dim=1)
        has_anchor = (~rc_anchor_mask).any(dim=1)
        is_active = has_cand & has_anchor

        if not bool(is_active.any().item()):

            return edit_logits

        active_idx = is_active.nonzero(as_tuple=False).squeeze(-1)
        h_dense_a = h_dense[active_idx]
        cand_mask_a = cand_mask[active_idx]
        rc_anchor_a = rc_anchor_emb[active_idx]
        rc_mask_a = rc_anchor_mask[active_idx]
        edge_padding_a = ~cand_mask_a

        out = h_dense_a
        for layer in self.layers:
            out = layer(
                out, rc_anchor_a,
                anchor_padding_mask=rc_mask_a,
                edge_padding_mask=edge_padding_a,
            )
        out = self.final_norm(out)
        edit_logits_dense = self.edit_classifier(out)

        active_edges_mask = torch.isin(edge_mol_ids, active_idx)
        edit_logits[active_edges_mask] = edit_logits_dense[cand_mask_a].to(edit_logits.dtype)

        return edit_logits

if __name__ == "__main__":
    print("=" * 60)
    print("§64 v34_E1 DETREditHead smoke test")
    print("=" * 60)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hidden = 512
    n_classes = 7
    K_max_rc = 10
    B = 3

    cand_counts = [12, 50, 0]
    K_total = sum(cand_counts)
    edge_mol_ids = torch.tensor(
        [b for b, k in enumerate(cand_counts) for _ in range(k)],
        dtype=torch.long, device=device,
    )
    h_edge = torch.randn(K_total, hidden, device=device, requires_grad=True)

    rc_anchor_emb = torch.randn(B, K_max_rc, hidden, device=device, requires_grad=True)
    rc_anchor_mask = torch.ones(B, K_max_rc, dtype=torch.bool, device=device)
    rc_anchor_mask[0, :3] = False
    rc_anchor_mask[1, :5] = False

    head = DETREditHead(
        hidden_dim=hidden, n_heads=8, n_decoder_layers=2, ffn_dim=2048,
        dropout=0.1, n_classes=n_classes,
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"DETREditHead trainable params: {n_params:,} (~{n_params / 1e6:.2f}M)")

    edit_logits = head(h_edge, edge_mol_ids, rc_anchor_emb, rc_anchor_mask)
    print(f"\nForward output:")
    print(f"  edit_logits.shape = {tuple(edit_logits.shape)}  [expected ({K_total}, {n_classes})]")
    assert edit_logits.shape == (K_total, n_classes), "shape mismatch"
    assert not torch.isnan(edit_logits).any(), "NaN in edit_logits"
    assert not torch.isinf(edit_logits).any(), "Inf in edit_logits"

    target = torch.randint(0, n_classes, (K_total,), device=device)
    loss = F.cross_entropy(edit_logits, target)
    loss.backward()
    grad_count = sum(1 for p in head.parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for _ in head.parameters())
    print(f"\nGradient flow: {grad_count}/{total_params} param tensors have nonzero grad")
    assert grad_count == total_params, f"missing grad on {total_params - grad_count} tensors"

    print(f"\n--- init_from_v29 (MLP structure copy) ---")
    fake_v29_sd = {
        "mlp.0.weight": torch.randn(256, hidden, device=device),
        "mlp.0.bias":   torch.randn(256, device=device),
        "mlp.1.weight": torch.randn(256, device=device),
        "mlp.1.bias":   torch.randn(256, device=device),
        "mlp.4.weight": torch.randn(n_classes, 256, device=device),
        "mlp.4.bias":   torch.randn(n_classes, device=device),
    }
    n_copied = head.init_edit_classifier_from_v29(fake_v29_sd)
    assert n_copied == 6, f"expected 6 tensors copied, got {n_copied}"

    assert torch.allclose(head.edit_classifier[0].weight.data, fake_v29_sd["mlp.0.weight"])
    assert torch.allclose(head.edit_classifier[0].bias.data, fake_v29_sd["mlp.0.bias"])
    assert torch.allclose(head.edit_classifier[1].weight.data, fake_v29_sd["mlp.1.weight"])
    assert torch.allclose(head.edit_classifier[1].bias.data, fake_v29_sd["mlp.1.bias"])
    assert torch.allclose(head.edit_classifier[4].weight.data, fake_v29_sd["mlp.4.weight"])
    assert torch.allclose(head.edit_classifier[4].bias.data, fake_v29_sd["mlp.4.bias"])
    print(f"  6/6 tensors copied (Linear×2 + LayerNorm×2): PASS")

    print(f"\n--- K=0 全 batch ---")
    empty_h = torch.zeros(0, hidden, device=device)
    empty_mol_ids = torch.zeros(0, dtype=torch.long, device=device)
    empty_logits = head(empty_h, empty_mol_ids, rc_anchor_emb, rc_anchor_mask)
    assert empty_logits.shape == (0, n_classes)
    print(f"  K=0 output shape={tuple(empty_logits.shape)}: PASS")

    print(f"\n--- 单 cand edge ---")
    single_h = torch.randn(1, hidden, device=device)
    single_mol_ids = torch.tensor([0], dtype=torch.long, device=device)
    single_logits = head(single_h, single_mol_ids, rc_anchor_emb[:1], rc_anchor_mask[:1])
    assert single_logits.shape == (1, n_classes)
    assert not torch.isnan(single_logits).any()
    print(f"  1-cand output shape={tuple(single_logits.shape)}: PASS")

    print(f"\n--- rc_anchor_mask 全 padding (defensive) ---")
    full_pad_mask = torch.ones(B, K_max_rc, dtype=torch.bool, device=device)
    defensive_logits = head(h_edge, edge_mol_ids, rc_anchor_emb, full_pad_mask)
    assert defensive_logits.shape == (K_total, n_classes)
    assert (defensive_logits == 0).all(), "expected all-zero when全 padding"
    print(f"  全 padding 防御 output all-zero: PASS")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
