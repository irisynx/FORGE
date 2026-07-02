"""推理侧化学约束解码 v0 (路线 B: 被动 H 平衡)

设计文档: analysis_reports/inference_constrained_decoding_v0/design.md

核心算法 (单分子):
  1. 候选边按 max_{c≠0} softmax 置信度降序
  2. 顺序应用最高置信度键, 每应用一条触发内层修复
  3. 内层修复: 对违规原子查邻接候选边的最高置信度键, 应用; 若无可用 → H 吸收 / 回滚
  4. 终止: 应用满 max_loops 或 conf < tau_stop

输入数据全部 CPU numpy/torch, 无 GPU 依赖.
"""
import heapq

import numpy as np
import torch

CLASS_TO_DELTA = np.array([0, 1, -1, 2, -2, 3, -3], dtype=np.int8)

DELTA_TO_CLASS = {0: 0, 1: 1, -1: 2, 2: 3, -2: 4, 3: 5, -3: 6}

def find_cls_step(cur_cls: int, direction: int):
    """delta + direction → 新 cls. direction ∈ {-1, +1}. 返回 cls or None."""
    cur_delta = int(CLASS_TO_DELTA[cur_cls])
    new_delta = cur_delta + direction
    if new_delta < -3 or new_delta > 3:
        return None
    return DELTA_TO_CLASS[new_delta]

STD_VALENCE = {
    1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1,
    14: 4, 15: 5, 16: 6, 17: 1,
    33: 5, 34: 6, 35: 1, 52: 6, 53: 1,
}

LENIENT_Z = frozenset({
    5,
    11, 12, 13,
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    50, 51,
    55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83,
})

def get_max_h(z, charge):
    if z in LENIENT_Z:
        return 99
    base = STD_VALENCE.get(int(z), 4)
    return max(0, base + int(charge))

def is_lenient(z):
    return int(z) in LENIENT_Z

class MolDecoder:
    """单分子贪心约束解码."""

    def __init__(self, atom_z, atom_h_before, atom_charge,
                 atom_total_degree_before,
                 reactant_bond_dict, cand_edges, edit_logits,
                 main_mask, hyperparams):
        self.n_atoms = len(atom_z)
        self.atom_z = np.asarray(atom_z, dtype=np.int32)
        self.atom_h = np.asarray(atom_h_before, dtype=np.int32).copy()
        self.atom_charge = np.asarray(atom_charge, dtype=np.int32)

        self.heavy_deg = (
            np.asarray(atom_total_degree_before, dtype=np.int32)
            - np.asarray(atom_h_before, dtype=np.int32)
        ).copy()
        self.bond_dict = dict(reactant_bond_dict)
        self.cand_edges = np.asarray(cand_edges, dtype=np.int32)
        self.k = int(self.cand_edges.shape[0])

        logits_t = torch.as_tensor(edit_logits, dtype=torch.float32)
        probs = torch.softmax(logits_t, dim=-1).numpy()
        self.probs = probs

        if self.k > 0:
            self.argmax_class = probs.argmax(1).astype(np.int8)
            nonzero_probs = probs[:, 1:]
            self.best_nonzero_class = (nonzero_probs.argmax(1) + 1).astype(np.int8)
            self.best_nonzero_conf = nonzero_probs.max(1).astype(np.float32)
            argmax_conf = probs[np.arange(self.k), self.argmax_class].astype(np.float32)
            apply_eligible = self.argmax_class > 0
            self.apply_class = np.where(apply_eligible, self.argmax_class, 0).astype(np.int8)
            self.apply_conf = np.where(apply_eligible, argmax_conf, -1.0).astype(np.float32)
        else:
            self.argmax_class = np.zeros(0, dtype=np.int8)
            self.best_nonzero_class = np.zeros(0, dtype=np.int8)
            self.best_nonzero_conf = np.zeros(0, dtype=np.float32)
            self.apply_class = np.zeros(0, dtype=np.int8)
            self.apply_conf = np.zeros(0, dtype=np.float32)

        self.main_mask = np.asarray(main_mask, dtype=np.bool_)
        self.hp = hyperparams

        self.delta_pred = np.zeros(self.k, dtype=np.int64)
        self.applied = set()
        self.visited = set()

        self.atom_to_edges = [[] for _ in range(self.n_atoms)]
        for ei in range(self.k):
            a = int(self.cand_edges[ei, 0])
            b = int(self.cand_edges[ei, 1])
            self.atom_to_edges[a].append(ei)
            self.atom_to_edges[b].append(ei)

        self.diag = {
            "n_apply": 0, "n_repair": 0, "n_h_absorb": 0,
            "n_rollback": 0, "n_cancel": 0, "n_cls_step": 0,
            "applied_class_ranks": [],
            "applied_edges": [],
        }

    def _snapshot_state(self):
        """Beam search: 捕获 MolDecoder 可变状态. 与 _rollback 内嵌捕获互补.

        包含 delta_pred / applied / visited / atom_h / heavy_deg / diag.
        diag 用 deepcopy 因含 applied_edges/applied_class_ranks list.
        """
        import copy
        return {
            "delta_pred": self.delta_pred.copy(),
            "applied": set(self.applied),
            "visited": set(self.visited),
            "atom_h": self.atom_h.copy(),
            "heavy_deg": self.heavy_deg.copy(),
            "diag": copy.deepcopy(self.diag),
        }

    def _restore_state(self, snap):
        """Beam search: 还原 MolDecoder 状态到 snapshot 时刻."""
        import copy
        self.delta_pred = snap["delta_pred"].copy()
        self.applied = set(snap["applied"])
        self.visited = set(snap["visited"])
        self.atom_h = snap["atom_h"].copy()
        self.heavy_deg = snap["heavy_deg"].copy()
        self.diag = copy.deepcopy(snap["diag"])

    def decode_beam(self, B, max_loops):
        """Beam search over candidate edge applications. B=1 ≡ greedy decode().

        每步对每个 beam 节点取 top-B 未 visited cand edges 逐个 apply (with repair),
        新 beam 大小最多 B*B, prune 到 top-B by joint log prob.
        Final pick = highest log_prob beam, 然后跑 cls_step post-process.

        Failed apply 后保留 rolled-back state (含 visited 更新), 不丢节点 — 与 greedy
        ptr 前进语义对齐.
        """

        if self.k > 0:
            self.delta_pred[:] = self.argmax_class
            self._init_h_from_argmax()
        if self.k == 0 or max_loops == 0:
            return self.delta_pred.copy(), dict(self.diag)

        order = np.argsort(-self.apply_conf)

        init_snap = self._snapshot_state()
        beam = [(init_snap, 0.0)]

        for _step in range(max_loops):
            new_beam = []
            for snap, log_p in beam:
                self._restore_state(snap)

                top_cands = []
                for idx in range(self.k):
                    cand = int(order[idx])
                    cls = int(self.apply_class[cand])
                    if cls == 0 or self.apply_conf[cand] < 0:
                        continue
                    if (cand, cls) in self.visited:
                        continue
                    conf = float(self.apply_conf[cand])
                    if conf < self.hp["tau_stop"]:
                        break
                    if conf < self.hp["tau_apply"]:
                        break
                    top_cands.append((cand, cls, conf))
                    if len(top_cands) >= B:
                        break

                if not top_cands:

                    new_beam.append((snap, log_p))
                    continue

                for cand, cls, conf in top_cands:
                    self._restore_state(snap)
                    self.visited.add((cand, cls))
                    self.applied.add(cand)
                    ok = self._try_repair_after_apply(cand)
                    if ok:
                        self.diag["n_apply"] += 1
                        self.diag["applied_edges"].append(int(cand))
                        new_snap = self._snapshot_state()

                        new_log_p = log_p + float(np.log(max(conf, 1e-8)))
                        new_beam.append((new_snap, new_log_p))
                    else:

                        new_snap = self._snapshot_state()
                        new_beam.append((new_snap, log_p))

            new_beam.sort(key=lambda x: -x[1])
            beam = new_beam[:B]
            if not beam:
                break

        if beam:
            best_snap, _ = beam[0]
            self._restore_state(best_snap)

        if self.hp.get("enable_cls_step", True):
            self._post_decode_cls_step()

        return self.delta_pred.copy(), dict(self.diag)

    def _full_loglik(self, delta_pred):
        """全分子联合 log-likelihood: Σ_e log p(delta_pred[e]).
        所有候选解在相同 k 项上比较, 无 beam apply-only log_p 的长度偏置."""
        if self.k == 0:
            return 0.0
        p = self.probs[np.arange(self.k), np.asarray(delta_pred, dtype=np.int64)]
        return float(np.sum(np.log(np.clip(p, 1e-12, None))))

    def decode_topk(self, B, max_loops, K):
        """返回 (result, n_distinct):
          result = K 个 (delta_pred, score), 长度恒 = K (不足用最后一个 padding);
          n_distinct = padding 前的互异解数.

        多样性来自 **逐边 7 类分布的联合 N-best 枚举**, 不是 apply 顺序 (后者 distinct 恒 1).
        rank-1 = greedy decode() (= 生产 baseline, 零回退); ranks 2..K = 从 argmax 出发,
        对 margin (logp_best - logp_2nd) 最小的边逐个翻到次优类, 用 min-heap 按联合 LL 降序枚举.
        含 class-0↔非零 翻转 (= 开/关一条边). B 当作可翻转边数 T 的上限.
        """
        if self.k == 0:
            return [(self.delta_pred.copy(), 0.0) for _ in range(max(K, 1))], 1

        anchor_dp, _ = self.decode(max_loops)
        anchor_score = self._full_loglik(anchor_dp)

        n_cls = self.probs.shape[1]
        logp = np.log(np.clip(self.probs, 1e-12, None))
        sorted_cls = np.argsort(-self.probs, axis=1)
        base_cls = sorted_cls[:, 0].astype(np.int64)
        ar = np.arange(self.k)
        margin = logp[ar, sorted_cls[:, 0]] - logp[ar, sorted_cls[:, 1]]

        T = min(int(B) if B and B > 0 else 2 * K, self.k)
        T = max(T, 1)
        flip_edges = np.argsort(margin)[:T]
        logp_flip = logp[flip_edges]
        scls_flip = sorted_cls[flip_edges]

        M = max(4 * K, 12)
        start = (0,) * T
        heap = [(0.0, start)]
        visited = {start}
        raw = []
        while heap and len(raw) < M:
            cost, ranks = heapq.heappop(heap)
            dp = base_cls.copy()
            for ti in range(T):
                r = ranks[ti]
                if r > 0:
                    dp[flip_edges[ti]] = scls_flip[ti, r]
            raw.append((cost, dp))
            for ti in range(T):
                r = ranks[ti] + 1
                if r >= n_cls:
                    continue
                nr = ranks[:ti] + (r,) + ranks[ti + 1:]
                if nr in visited:
                    continue
                visited.add(nr)

                new_cost = cost + (logp_flip[ti, scls_flip[ti, ranks[ti]]]
                                   - logp_flip[ti, scls_flip[ti, r]])
                heapq.heappush(heap, (new_cost, nr))

        seen = {anchor_dp.tobytes()}
        result = [(anchor_dp, anchor_score)]
        for cost, dp in raw:
            key = dp.tobytes()
            if key in seen:
                continue
            seen.add(key)
            result.append((dp, -float(cost)))
            if len(result) >= K:
                break
        n_distinct = len(result)
        while len(result) < K:
            result.append(result[-1])
        return result, n_distinct

    def decode(self, max_loops):

        if self.k > 0:
            self.delta_pred[:] = self.argmax_class
            self._init_h_from_argmax()
        if self.k == 0 or max_loops == 0:
            return self.delta_pred.copy(), dict(self.diag)

        order = np.argsort(-self.apply_conf)
        ptr = 0

        for _ in range(max_loops):
            ei = -1
            while ptr < self.k:
                cand = int(order[ptr])
                cls = int(self.apply_class[cand])
                if cls == 0 or self.apply_conf[cand] < 0:
                    ptr += 1
                    continue
                if (cand, cls) in self.visited:
                    ptr += 1
                    continue
                ei = cand
                break
            if ei < 0:
                break

            conf = float(self.apply_conf[ei])
            if conf < self.hp["tau_stop"]:
                break
            if conf < self.hp["tau_apply"]:
                break

            ptr += 1
            cls = int(self.apply_class[ei])

            self.visited.add((ei, cls))
            self.applied.add(ei)
            ok = self._try_repair_after_apply(ei)
            if ok:
                self.diag["n_apply"] += 1
                self.diag["applied_edges"].append(int(ei))

        if self.hp.get("enable_cls_step", True):
            self._post_decode_cls_step()

        return self.delta_pred.copy(), dict(self.diag)

    def _post_decode_cls_step(self):
        """对每个 main_mask 原子检查 net delta, 不平则 cls step 调整最低 cost incident 边.
        保守: 仅动 cls != 0 的边 (避免从 cls=0 制造 FP).
        v2: 加 |net| 阈值, 仅严重偏差才触发 (避免误调真有 net != 0 的反应)."""
        max_iters = self.hp.get("cls_step_max_iters", 3)
        max_cost = self.hp.get("cls_step_max_cost", 0.40)
        net_threshold = self.hp.get("cls_step_net_threshold", 2)
        for _ in range(max_iters):
            any_changed = False
            for i in range(self.n_atoms):
                if not self.main_mask[i]:
                    continue
                z = int(self.atom_z[i])
                if z in LENIENT_Z:
                    continue

                net = 0
                for ei in self.atom_to_edges[i]:
                    net += int(CLASS_TO_DELTA[int(self.delta_pred[ei])])
                if abs(net) < net_threshold:
                    continue
                direction = -1 if net > 0 else +1
                step_cands = []
                for ei in self.atom_to_edges[i]:
                    cur_cls = int(self.delta_pred[ei])
                    if cur_cls == 0:
                        continue
                    new_cls = find_cls_step(cur_cls, direction)
                    if new_cls is None or new_cls == cur_cls:
                        continue
                    if (ei, new_cls) in self.visited:
                        continue
                    p_cur = float(self.probs[ei, cur_cls])
                    p_new = float(self.probs[ei, new_cls])
                    cost = p_cur - p_new
                    step_cands.append((ei, new_cls, cost))
                step_cands.sort(key=lambda x: x[2])
                if step_cands and step_cands[0][2] < max_cost:
                    ei, new_cls, _ = step_cands[0]
                    old_cls = int(self.delta_pred[ei])
                    self.delta_pred[ei] = new_cls
                    self.applied.add(ei)
                    self.visited.add((ei, new_cls))
                    self._apply_state_change(ei, old_cls, new_cls)
                    self.diag["n_cls_step"] += 1
                    any_changed = True
            if not any_changed:
                break

    def _init_h_from_argmax(self):
        """根据 argmax_class 初始化 atom_h + heavy_deg (passive balance)."""
        for ei in range(self.k):
            cls = int(self.argmax_class[ei])
            if cls == 0:
                continue
            self._apply_state_change(ei, 0, cls)

    def _apply_state_change(self, ei, old_cls, new_cls):
        """edge ei 类别从 old_cls 切到 new_cls, 同步 atom_h + heavy_deg."""
        a = int(self.cand_edges[ei, 0])
        b = int(self.cand_edges[ei, 1])
        if a > b:
            a, b = b, a
        old_delta = int(CLASS_TO_DELTA[old_cls])
        new_delta = int(CLASS_TO_DELTA[new_cls])
        old_order = self.bond_dict.get((a, b), 0.0) + old_delta
        new_order = self.bond_dict.get((a, b), 0.0) + new_delta

        old_has = old_order > 0.5
        new_has = new_order > 0.5
        if not old_has and new_has:
            self.heavy_deg[a] += 1
            self.heavy_deg[b] += 1
        elif old_has and not new_has:
            self.heavy_deg[a] -= 1
            self.heavy_deg[b] -= 1

        d_diff = new_delta - old_delta
        self.atom_h[a] -= d_diff
        self.atom_h[b] -= d_diff

    def _try_repair_after_apply(self, trigger_ei):
        """apply 一条边后触发的内层修复. 现在 delta_pred 已更新, 检查 valence,
        若违规, 用 best_nonzero_conf 找邻接边修复.
        失败则回滚到 trigger_ei apply 之前的状态.
        """
        snap_delta = self.delta_pred.copy()
        snap_applied = set(self.applied)
        snap_visited = set(self.visited)
        snap_h = self.atom_h.copy()
        snap_heavy = self.heavy_deg.copy()

        for _ in range(self.hp["max_inner_iter"]):
            real_v = self._real_violators()
            if not real_v:
                return True

            i = real_v[0]
            cands = []
            for nei in self.atom_to_edges[i]:

                k_repair = int(self.best_nonzero_class[nei])
                if (nei, k_repair) in self.visited:
                    continue

                if int(self.delta_pred[nei]) == k_repair:
                    continue
                cands.append((nei, k_repair, float(self.best_nonzero_conf[nei])))
            cands.sort(key=lambda x: -x[2])

            if not cands or cands[0][2] < self.hp["tau_repair"]:

                if self.hp.get("enable_cancel", True):
                    cancel_cands = []
                    for nei in self.atom_to_edges[i]:
                        cls_n = int(self.delta_pred[nei])
                        if cls_n == 0:
                            continue
                        if (nei, 0) in self.visited:
                            continue
                        p0 = float(self.probs[nei, 0])
                        p_pred = float(self.probs[nei, cls_n])
                        margin = p_pred - p0
                        cancel_cands.append((nei, margin))

                    cancel_threshold = self.hp.get("cancel_margin_max", 0.30)
                    cancel_cands = [c for c in cancel_cands if c[1] < cancel_threshold]
                    cancel_cands.sort(key=lambda x: x[1])
                    if cancel_cands:
                        nei, _ = cancel_cands[0]
                        old_cls = int(self.delta_pred[nei])
                        self.delta_pred[nei] = 0
                        self.applied.discard(nei)
                        self.visited.add((nei, 0))
                        self._apply_state_change(nei, old_cls, 0)
                        self.diag["n_cancel"] += 1
                        continue
                if self._h_absorb_ok(i):
                    self.diag["n_h_absorb"] += 1
                    return True
                self._rollback(snap_delta, snap_applied, snap_visited, snap_h, snap_heavy)
                return False

            nei, k_new, _ = cands[0]
            old_cls = int(self.delta_pred[nei])
            self.delta_pred[nei] = k_new
            self.applied.add(nei)
            self.visited.add((nei, k_new))
            self._apply_state_change(nei, old_cls, k_new)
            self.diag["n_repair"] += 1

        if not self._real_violators():
            return True

        self._rollback(snap_delta, snap_applied, snap_visited, snap_h, snap_heavy)
        return False

    def _rollback(self, snap_delta, snap_applied, snap_visited, snap_h, snap_heavy):
        self.delta_pred = snap_delta
        self.applied = snap_applied
        self.visited = snap_visited
        self.atom_h = snap_h
        self.heavy_deg = snap_heavy
        self.diag["n_rollback"] += 1

    def _update_h(self, ei, cls):
        d = int(CLASS_TO_DELTA[cls])
        a = int(self.cand_edges[ei, 0])
        b = int(self.cand_edges[ei, 1])
        self.atom_h[a] -= d
        self.atom_h[b] -= d

    def _swap_h(self, ei, old_cls, new_cls):
        """edge ei 的 class 从 old_cls 换到 new_cls, 更新 atom_h 增量."""
        d_old = int(CLASS_TO_DELTA[old_cls])
        d_new = int(CLASS_TO_DELTA[new_cls])
        d_diff = d_new - d_old
        a = int(self.cand_edges[ei, 0])
        b = int(self.cand_edges[ei, 1])
        self.atom_h[a] -= d_diff
        self.atom_h[b] -= d_diff

    def _real_violators(self):
        """检测违规原子: H<0 / 总价超标 / heavy_deg 超标."""
        out = []
        for i in range(self.n_atoms):
            if not self.main_mask[i]:
                continue
            z = int(self.atom_z[i])
            if z in LENIENT_Z:
                continue
            h = int(self.atom_h[i])
            heavy = int(self.heavy_deg[i])
            charge = int(self.atom_charge[i])
            max_v = STD_VALENCE.get(z, 4) + charge
            if h < 0:
                out.append(i)
                continue
            if heavy + h > max_v:
                out.append(i)
                continue
            if heavy > max_v:
                out.append(i)
        return out

    def _h_absorb_ok(self, i):
        z = int(self.atom_z[i])
        if z in LENIENT_Z:
            return True
        h = int(self.atom_h[i])
        heavy = int(self.heavy_deg[i])
        charge = int(self.atom_charge[i])
        max_v = STD_VALENCE.get(z, 4) + charge
        if h < 0:
            return False
        if heavy + h > max_v:
            return False
        if heavy > max_v:
            return False
        return True

def decode_batch(outputs, batch, T, hyperparams, max_loops, beam_size=1):
    """对 batch 内每个分子调用 MolDecoder, 返回 batch 级 edit_pred + 诊断列表.

    beam_size=1 ≡ greedy decode (back-compat). beam_size>=2 启用 beam search
    (§63.2.3 E 路线), top-B candidates per step, prune by joint log prob.
    """
    edit_logits = outputs["edit_logits"].float().cpu().numpy()
    cand_edges_list = outputs["cand_edges"]
    batch_idx = batch.batch.cpu().numpy()
    if len(batch_idx) == 0:
        return torch.zeros(0, dtype=torch.long), []
    batch_size = int(batch_idx.max()) + 1

    x = batch.x.cpu().numpy()
    atom_z_global = x[:, 0]
    atom_degree_global = x[:, 1]
    atom_h_global = x[:, 7]
    atom_charge_global = x[:, 2].astype(np.int32) - 5

    edge_index = batch.edge_index.cpu().numpy()
    edge_attr = batch.edge_attr.cpu().numpy().astype(np.float32)
    y_delta_list = batch.y_delta_list

    node_counts = np.bincount(batch_idx, minlength=batch_size)
    node_offsets = np.concatenate([[0], np.cumsum(node_counts)[:-1]])

    edge_counts = [int(e.size(0)) for e in cand_edges_list]
    edge_offsets = np.concatenate([[0], np.cumsum(edge_counts)[:-1]]) if edge_counts else np.array([0])

    edit_pred_full = np.zeros(sum(edge_counts), dtype=np.int64)
    diag_per_mol = []

    for mi in range(batch_size):
        n = int(node_counts[mi])
        start = int(node_offsets[mi])
        edge_start = int(edge_offsets[mi])
        k = int(edge_counts[mi]) if mi < len(edge_counts) else 0

        if k == 0:
            diag_per_mol.append({
                "n_apply": 0, "n_repair": 0, "n_h_absorb": 0,
                "n_rollback": 0, "k": 0, "applied_edges": [],
            })
            continue

        emask = (edge_index[0] >= start) & (edge_index[0] < start + n)
        local_ei = edge_index[:, emask] - start
        local_ea = edge_attr[emask]
        bond_dict = {}
        for ek in range(local_ei.shape[1]):
            i = int(local_ei[0, ek])
            j = int(local_ei[1, ek])
            if i < j:
                bond_dict[(i, j)] = float(local_ea[ek])

        y_delta = y_delta_list[mi]
        if not isinstance(y_delta, torch.Tensor):
            y_delta = torch.tensor(y_delta)
        main_mask = T.compute_main_product_atoms(
            torch.from_numpy(local_ei.astype(np.int64)),
            torch.from_numpy(local_ea),
            y_delta.cpu(),
            n,
        ).cpu().numpy().astype(bool)

        local_z = atom_z_global[start:start + n]
        local_h = atom_h_global[start:start + n]
        local_charge = atom_charge_global[start:start + n]
        local_degree = atom_degree_global[start:start + n]
        cand_edges_local = cand_edges_list[mi].cpu().numpy()
        edit_logits_mol = edit_logits[edge_start:edge_start + k]

        decoder = MolDecoder(
            atom_z=local_z,
            atom_h_before=local_h,
            atom_charge=local_charge,
            atom_total_degree_before=local_degree,
            reactant_bond_dict=bond_dict,
            cand_edges=cand_edges_local,
            edit_logits=edit_logits_mol,
            main_mask=main_mask,
            hyperparams=hyperparams,
        )
        if beam_size > 1:
            delta_pred, diag = decoder.decode_beam(beam_size, max_loops)
        else:
            delta_pred, diag = decoder.decode(max_loops)
        edit_pred_full[edge_start:edge_start + k] = delta_pred
        diag["k"] = k
        diag_per_mol.append(diag)

    return torch.from_numpy(edit_pred_full), diag_per_mol

def decode_batch_topk(outputs, batch, T, hyperparams, max_loops, B, K):
    """top-K 版 decode_batch. 返回 (edit_pred_ranks, diag_per_mol):
       edit_pred_ranks = K 个 full-batch edit_pred 张量 (rank 1..K, 同 decode_batch 的边布局);
       diag_per_mol[mi] = {'k', 'n_distinct'} (该分子互异解数, k==0 分子 n_distinct=1).
       某分子互异解 < rank 时该 rank 槽位重复其 best 解 (对 any-match 语义无害)."""
    edit_logits = outputs["edit_logits"].float().cpu().numpy()
    cand_edges_list = outputs["cand_edges"]
    batch_idx = batch.batch.cpu().numpy()
    if len(batch_idx) == 0:
        return [torch.zeros(0, dtype=torch.long) for _ in range(K)], []
    batch_size = int(batch_idx.max()) + 1

    x = batch.x.cpu().numpy()
    atom_z_global = x[:, 0]
    atom_degree_global = x[:, 1]
    atom_h_global = x[:, 7]
    atom_charge_global = x[:, 2].astype(np.int32) - 5

    edge_index = batch.edge_index.cpu().numpy()
    edge_attr = batch.edge_attr.cpu().numpy().astype(np.float32)
    y_delta_list = batch.y_delta_list

    node_counts = np.bincount(batch_idx, minlength=batch_size)
    node_offsets = np.concatenate([[0], np.cumsum(node_counts)[:-1]])

    edge_counts = [int(e.size(0)) for e in cand_edges_list]
    edge_offsets = np.concatenate([[0], np.cumsum(edge_counts)[:-1]]) if edge_counts else np.array([0])

    total_edges = sum(edge_counts)
    edit_pred_ranks = [np.zeros(total_edges, dtype=np.int64) for _ in range(K)]
    diag_per_mol = []

    for mi in range(batch_size):
        n = int(node_counts[mi])
        start = int(node_offsets[mi])
        edge_start = int(edge_offsets[mi])
        k = int(edge_counts[mi]) if mi < len(edge_counts) else 0

        if k == 0:
            diag_per_mol.append({"k": 0, "n_distinct": 1})
            continue

        emask = (edge_index[0] >= start) & (edge_index[0] < start + n)
        local_ei = edge_index[:, emask] - start
        local_ea = edge_attr[emask]
        bond_dict = {}
        for ek in range(local_ei.shape[1]):
            i = int(local_ei[0, ek])
            j = int(local_ei[1, ek])
            if i < j:
                bond_dict[(i, j)] = float(local_ea[ek])

        y_delta = y_delta_list[mi]
        if not isinstance(y_delta, torch.Tensor):
            y_delta = torch.tensor(y_delta)
        main_mask = T.compute_main_product_atoms(
            torch.from_numpy(local_ei.astype(np.int64)),
            torch.from_numpy(local_ea),
            y_delta.cpu(),
            n,
        ).cpu().numpy().astype(bool)

        decoder = MolDecoder(
            atom_z=atom_z_global[start:start + n],
            atom_h_before=atom_h_global[start:start + n],
            atom_charge=atom_charge_global[start:start + n],
            atom_total_degree_before=atom_degree_global[start:start + n],
            reactant_bond_dict=bond_dict,
            cand_edges=cand_edges_list[mi].cpu().numpy(),
            edit_logits=edit_logits[edge_start:edge_start + k],
            main_mask=main_mask,
            hyperparams=hyperparams,
        )
        topk, n_distinct = decoder.decode_topk(B, max_loops, K)
        for r in range(K):
            dp, _ = topk[r]
            edit_pred_ranks[r][edge_start:edge_start + k] = dp
        diag_per_mol.append({"k": k, "n_distinct": int(n_distinct)})

    return [torch.from_numpy(a) for a in edit_pred_ranks], diag_per_mol

DEFAULT_HYPERPARAMS = {
    "tau_apply": 0.40,
    "tau_repair": 0.25,
    "tau_stop": 0.30,
    "max_inner_iter": 5,
    "enable_cancel": True,
    "cancel_margin_max": 0.30,

    "enable_cls_step": False,
    "cls_step_max_cost": 0.05,
    "cls_step_max_iters": 3,
    "cls_step_net_threshold": 2,
}
