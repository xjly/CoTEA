"""Protocol Audit 1 & 2:
1. Row-wise Cosine & Row-wise CSLS-5 (without Sinkhorn) vs. Balanced Sinkhorn on Open Pools (WUXI-TEMP 5483, DICEWS 9537, BETA-30%).
2. Shared Relation Labels vs. Strictly Disjoint Relation Namespace on DICEWS-1K for both baselines and MSEA-Align.
"""
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT_EXP = Path(__file__).resolve().parents[2]
RCTEA_ROOT = ROOT_EXP / 'third_party/rctea/dicews'
MGTEA_SRC = ROOT_EXP / 'third_party/mgtea'
sys.path.insert(0, str(MGTEA_SRC))
sys.path.insert(0, str(RCTEA_ROOT))

from reproduction.benchmark_retrieval import balance_inplace
from reproduction.data import read_vocab, load_pairs, save_json
from reproduction.train import average_ranks


@torch.inference_mode()
def score_open_pool_all_modes(raw_sim_n_by_m, query_pairs, pool_rows, pool_cols, n, total_e, is_already_csls=False):
    """Return Row-wise Raw/Cosine, Row-wise CSLS-5 (no Sinkhorn), and CSLS-5 + Sinkhorn200 on the specified pool."""
    sub_raw = raw_sim_n_by_m[pool_rows][:, pool_cols - n].double()
    if is_already_csls:
        sub_csls = sub_raw.clone()
    else:
        rx = sub_raw.topk(5, dim=1).values.mean(1, keepdim=True)
        ry = sub_raw.topk(5, dim=0).values.mean(0, keepdim=True)
        sub_csls = 2.0 * sub_raw - rx - ry

    sub_sink = sub_csls.clone()
    u, v = balance_inplace(sub_sink, temperature=0.05, iterations=200, block=256)
    sub_sink.add_(u[:, None]).add_(v[None, :])

    rowmap = np.full(total_e, -1, dtype=np.int64)
    rowmap[pool_rows] = np.arange(len(pool_rows))
    colmap = np.full(total_e, -1, dtype=np.int64)
    colmap[pool_cols] = np.arange(len(pool_cols))
    q = torch.as_tensor(rowmap[query_pairs[:, 0]], device='cuda')
    g = torch.as_tensor(colmap[query_pairs[:, 1]], device='cuda')

    def m_fn(mat):
        r = average_ranks(mat[q], g).cpu().numpy()
        return dict(
            mrr=float((1.0 / r).mean()),
            hits1=float((r <= 1.0).mean()),
            hits10=float((r <= 10.0).mean()),
            queries=int(len(r)),
        )

    return dict(
        rowwise_raw_or_cosine=m_fn(sub_raw),
        rowwise_csls5_nosinkhorn=m_fn(sub_csls),
        csls5_sinkhorn200=m_fn(sub_sink),
    )


@torch.inference_mode()
def eval_wuxi_nosinkhorn_audit():
    """Audit all methods on WUXI-TEMP open pool (eval_5483 and full_6483) with and without Sinkhorn."""
    raw_dir = ROOT_EXP / 'datasets/WUXI-TEMP'
    suite_dir = ROOT_EXP / 'experiments/wuxi_suite_20261108'
    cache = suite_dir / 'cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m
    total_r = protocol['shared_relations']
    total_t = protocol['time_tokens']

    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train = splits['train']
        reference = splits['reference']
        candidates_eval = splits['candidates_eval']
    with np.load(cache / 'events.npz') as ev_npz:
        events = ev_npz['events']
        lengths = ev_npz['lengths']

    ev = [read_vocab(raw_dir / f'ent_ids_{i}') for i in (1, 2)]
    raw_ids = [sorted(v) for v in ev]
    indices = [{v: j + off for j, v in enumerate(ids)} for ids, off in zip(raw_ids, (0, n))]
    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    ri = {label: i for i, label in enumerate(protocol['relation_names'])}
    ti = {label: int(k) - 1 for k, label in protocol['time_labels'].items()}

    # 1. LightEA & LightTEA
    ee_edges, er_edges, re_edges, et_edges, te_edges = Counter(), Counter(), Counter(), Counter(), Counter()
    entity_time_sets = [set() for _ in range(total_e)]
    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, start, _ = line.split('\t')
            h, t = indices[g][int(h_raw)], indices[g][int(t_raw)]
            r = ri[rv[g][int(r_raw)]]
            ee_edges[(h, t)] += 1.0
            ee_edges[(t, h)] += 1.0
            er_edges[(h, 2 * r)] += 1.0
            er_edges[(t, 2 * r + 1)] += 1.0
            re_edges[(2 * r, h)] += 1.0
            re_edges[(2 * r + 1, t)] += 1.0
            if start != '~':
                tid = ti[start]
                et_edges[(h, tid)] += 1.0
                et_edges[(t, tid)] += 1.0
                te_edges[(tid, h)] += 1.0
                te_edges[(tid, t)] += 1.0
                entity_time_sets[h].add(tid)
                entity_time_sets[t].add(tid)

    def make_sparse_row_norm(edge_counter, shape):
        rows, cols, vals = [], [], []
        row_sums = defaultdict(float)
        for (r_i, c_i), v in edge_counter.items():
            w = math.log1p(v)
            rows.append(r_i)
            cols.append(c_i)
            vals.append(w)
            row_sums[r_i] += w
        vals = [v / max(row_sums[r_i], 1e-12) for r_i, v in zip(rows, vals)]
        idx = torch.tensor([rows, cols], dtype=torch.long, device='cuda')
        val_t = torch.tensor(vals, dtype=torch.float32, device='cuda')
        return torch.sparse_coo_tensor(idx, val_t, size=shape, device='cuda').coalesce()

    A_EE = make_sparse_row_norm(ee_edges, (total_e, total_e))
    A_ER = make_sparse_row_norm(er_edges, (total_e, 2 * total_r))
    A_RE = make_sparse_row_norm(re_edges, (2 * total_r, total_e))
    A_ET = make_sparse_row_norm(et_edges, (total_e, total_t))
    A_TE = make_sparse_row_norm(te_edges, (total_t, total_e))

    gen = torch.Generator(device='cuda').manual_seed(2026)
    seed_vecs = F.normalize(torch.randn(len(train), 300, generator=gen, device='cuda'), dim=-1)
    L_e0 = torch.zeros(total_e, 300, device='cuda')
    for idx_s, (u, v) in enumerate(train):
        L_e0[int(u)] = seed_vecs[idx_s]
        L_e0[int(v)] = seed_vecs[idx_s]

    L_e_rel, L_r, curr_e = [L_e0], torch.zeros(2 * total_r, 300, device='cuda'), L_e0
    for _ in range(2):
        curr_e, L_r = F.normalize(torch.sparse.mm(A_EE, curr_e) + torch.sparse.mm(A_ER, L_r), dim=-1), F.normalize(torch.sparse.mm(A_RE, curr_e), dim=-1)
        L_e_rel.append(curr_e)
    H_rel = F.normalize(torch.cat(L_e_rel, dim=-1), dim=-1)

    L_e_time, L_t, curr_et = [L_e0], torch.zeros(total_t, 300, device='cuda'), L_e0
    for _ in range(2):
        curr_et, L_t = F.normalize(torch.sparse.mm(A_EE, curr_et) + torch.sparse.mm(A_ET, L_t), dim=-1), F.normalize(torch.sparse.mm(A_TE, curr_et), dim=-1)
        L_e_time.append(curr_et)
    H_time = F.normalize(torch.cat(L_e_time, dim=-1), dim=-1)

    inc = torch.zeros(total_e, total_t, dtype=torch.float32, device='cuda')
    for eid, tset in enumerate(entity_time_sets):
        if tset:
            inc[eid, list(tset)] = 1.0
    inter = inc[:n] @ inc[n:].T
    denom = inc[:n].sum(1, keepdim=True) + inc[n:].sum(1)[None, :]
    S_time_dice = torch.where(denom > 0, 2.0 * inter / denom.clamp_min(1.0), torch.zeros_like(inter))

    sim_lightea = H_rel[:n] @ H_rel[n:].T
    H_ltea = F.normalize(0.7 * H_rel + 0.3 * H_time, dim=-1)
    sim_lighttea = 0.7 * (H_ltea[:n] @ H_ltea[n:].T) + 0.3 * S_time_dice

    # Load saved HTEA, RCTEA, MGTEA embeddings on WUXI-TEMP
    feat_htea = torch.from_numpy(np.load(suite_dir / 'htea_wuxi_embeddings.npy')).cuda()
    sim_htea = feat_htea[:n] @ feat_htea[n:].T

    feat_rctea = torch.from_numpy(np.load(suite_dir / 'rctea_wuxi_embeddings.npy')).cuda()
    sim_rctea = feat_rctea[:n] @ feat_rctea[n:].T

    feat_mgtea = torch.from_numpy(np.load(suite_dir / 'mgtea_wuxi_embeddings.npy')).cuda()
    sim_mgtea = (feat_mgtea[:n] @ feat_mgtea[n:].T + 0.5 * S_time_dice) / 1.5

    # Reconstruct MSEA-Align composite CSLS matrix on WUXI-TEMP
    sys.path.insert(0, str(suite_dir))
    from train_wuxi_suite import csls_matrix
    z_ht_full = np.load(suite_dir / 'runs/hier_time/final_full_vectors.npy')
    z_ht_notime = np.load(suite_dir / 'runs/hier_time/final_notime_vectors.npy')
    z_ht_coarse = np.load(suite_dir / 'runs/hier_time/final_coarse_vectors.npy')
    cf = csls_matrix(torch.from_numpy(z_ht_full[:n]).cuda(), torch.from_numpy(z_ht_full[n:]).cuda(), k=5)
    cs = csls_matrix(torch.from_numpy(z_ht_notime[:n]).cuda(), torch.from_numpy(z_ht_notime[n:]).cuda(), k=5)
    cc = csls_matrix(torch.from_numpy(z_ht_coarse[:n]).cuda(), torch.from_numpy(z_ht_coarse[n:]).cuda(), k=5)
    csls_neural_tri = (cf + 0.35 * cc + 0.15 * cs) / 1.50

    is_hub = lengths > 64
    node_deg = np.maximum(lengths.astype(np.float64), 1.0)
    topo_edges, all_idf_edges = Counter(), Counter()
    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, _, t_raw, _, _ = line.split('\t')
            h, t = indices[g][int(h_raw)], indices[g][int(t_raw)]
            if not is_hub[h] and not is_hub[t]:
                topo_edges[(h, t)] += 1.0
                topo_edges[(t, h)] += 1.0
            all_idf_edges[(h, t)] += 1.0 / math.sqrt(node_deg[h] * node_deg[t])
            all_idf_edges[(t, h)] += 1.0 / math.sqrt(node_deg[h] * node_deg[t])

    def build_sp(edges_cnt):
        rows, cols, vals = [], [], []
        rsum = defaultdict(float)
        for (u, v), w in edges_cnt.items():
            rows.append(u)
            cols.append(v)
            vals.append(w)
            rsum[u] += w
        vals = [w / max(rsum[u], 1e-12) for u, w in zip(rows, vals)]
        return torch.sparse_coo_tensor(torch.tensor([rows, cols], device='cuda'), torch.tensor(vals, dtype=torch.float32, device='cuda'), size=(total_e, total_e)).coalesce()

    A_topo = build_sp(topo_edges)
    A_idf = build_sp(all_idf_edges)
    gen_m = torch.Generator(device='cuda').manual_seed(11732026)
    rel_counts = np.load(suite_dir / 'profiles/seed_relation_counts.npy')
    obj_counts = np.load(suite_dir / 'profiles/seed_object_counts.npy')
    rel_coeff = torch.tensor(np.log1p(rel_counts), dtype=torch.float32, device='cuda')
    obj_coeff = torch.tensor(np.log1p(obj_counts), dtype=torch.float32, device='cuda')
    _ = torch.randn(2, len(train), 256, generator=gen_m, device='cuda')  # keep identical RNG sequence as test_wuxi_breakthrough.py
    seed_vecs_m = F.normalize(torch.randn(len(train), 512, generator=gen_m, device='cuda'), dim=-1)
    X0 = torch.zeros(total_e, 512, device='cuda')
    for idx_s, (u, v) in enumerate(train):
        X0[int(u)] = seed_vecs_m[idx_s]
        X0[int(v)] = seed_vecs_m[idx_s]
    topo_hops, idf_hops = [], []
    ct, ci = X0, X0
    for _ in range(3):
        ct = torch.sparse.mm(A_topo, ct)
        ci = torch.sparse.mm(A_idf, ci)
        topo_hops.append(F.normalize(ct, dim=-1))
        idf_hops.append(F.normalize(ci, dim=-1))
    Z_topo = F.normalize(topo_hops[0] + 0.7 * topo_hops[1] + 0.5 * topo_hops[2], dim=-1)
    Z_idf = F.normalize(idf_hops[0] + 0.7 * idf_hops[1] + 0.5 * idf_hops[2], dim=-1)
    csls_topo = csls_matrix(Z_topo[:n], Z_topo[n:], k=5)
    csls_idf = csls_matrix(Z_idf[:n], Z_idf[n:], k=5)

    rel_bind = F.normalize(torch.einsum('rks,ksd->rd', rel_coeff, torch.randn(2, len(train), 512, generator=gen_m, device='cuda')), dim=-1)
    hub_bind = F.normalize(torch.einsum('oks,ksd->od', obj_coeff, torch.randn(2, len(train), 512, generator=gen_m, device='cuda')), dim=-1)
    dir_bind = F.normalize(torch.randn(2, 512, generator=gen_m, device='cuda'), dim=-1)
    time_labels = protocol['time_labels']
    ym_vocab, y_vocab = {}, {}
    tok_to_ym = np.zeros(len(time_labels) + 1, dtype=np.int64)
    tok_to_y = np.zeros(len(time_labels) + 1, dtype=np.int64)
    tok_is_day = np.zeros(len(time_labels) + 1, dtype=bool)
    for k_str, lbl in time_labels.items():
        tok = int(k_str)
        y_str = lbl[:4]
        if y_str not in y_vocab:
            y_vocab[y_str] = len(y_vocab) + 1
        tok_to_y[tok] = y_vocab[y_str]
        if lbl[5:7] != '00':
            ym_str = lbl[:7]
            if ym_str not in ym_vocab:
                ym_vocab[ym_str] = len(ym_vocab) + 1
            tok_to_ym[tok] = ym_vocab[ym_str]
        if not lbl.endswith('00'):
            tok_is_day[tok] = True
    t_day_vecs = F.normalize(torch.randn(len(time_labels) + 1, 512, generator=gen_m, device='cuda'), dim=-1)
    t_day_vecs[0] = 0.0
    t_ym_vecs = F.normalize(torch.randn(len(ym_vocab) + 1, 512, generator=gen_m, device='cuda'), dim=-1)
    t_ym_vecs[0] = 0.0
    t_y_vecs = F.normalize(torch.randn(len(y_vocab) + 1, 512, generator=gen_m, device='cuda'), dim=-1)
    t_y_vecs[0] = 0.0
    ev_t = torch.as_tensor(events, dtype=torch.long, device='cuda')
    valid = ev_t[:, :, 0] > 0
    obj_id = (ev_t[:, :, 0] - 1).clamp_min(0)
    rel_id = ((ev_t[:, :, 1] - 1) // 2).clamp_min(0)
    rel_dir = (ev_t[:, :, 1] - 1).clamp_min(0) % 2
    start_tok = ev_t[:, :, 2]
    freq_w = torch.log1p(ev_t[:, :, 3].float()) * valid.float()
    sem_state = F.normalize(rel_bind[rel_id] + 0.5 * dir_bind[rel_dir] + hub_bind[obj_id], dim=-1)
    ym_tok = torch.as_tensor(tok_to_ym, device='cuda')[start_tok]
    y_tok = torch.as_tensor(tok_to_y, device='cuda')[start_tok]
    is_day_mask = torch.as_tensor(tok_is_day, device='cuda')[start_tok] & valid
    bound_day = ((sem_state * t_day_vecs[start_tok]) * (freq_w * is_day_mask.float())[:, :, None]).sum(1)
    bound_ym = ((sem_state * t_ym_vecs[ym_tok]) * (freq_w * (ym_tok > 0).float())[:, :, None]).sum(1)
    bound_y = ((sem_state * t_y_vecs[y_tok]) * (freq_w * (y_tok > 0).float())[:, :, None]).sum(1)
    bound_notime = (sem_state * freq_w[:, :, None]).sum(1)
    Z_coupled = F.normalize(1.0 * F.normalize(bound_day, dim=-1) + 0.7 * F.normalize(bound_ym, dim=-1) + 0.3 * F.normalize(bound_y, dim=-1) + 0.4 * F.normalize(bound_notime, dim=-1), dim=-1)
    Z_coupled_prop = F.normalize(Z_coupled + 0.5 * torch.sparse.mm(A_topo, Z_coupled), dim=-1)
    csls_coupled_prop = csls_matrix(Z_coupled_prop[:n], Z_coupled_prop[n:], k=5)

    msea_comp = csls_neural_tri + 0.8 * csls_coupled_prop + 0.6 * csls_topo + 0.4 * csls_idf

    wuxi_audit = {}
    for m_name, sim_mat, is_csls in [
        ('LightEA', sim_lightea, False),
        ('LightTEA', sim_lighttea, False),
        ('HTEA', sim_htea, False),
        ('RCTEA', sim_rctea, False),
        ('MGTEA', sim_mgtea, False),
        ('MSEA-Align', msea_comp, True),
    ]:
        wuxi_audit[m_name] = {
            'eval_5483': score_open_pool_all_modes(sim_mat, reference, reference[:, 0], candidates_eval, n, total_e, is_already_csls=is_csls),
            'closed_3872': score_open_pool_all_modes(sim_mat, reference, reference[:, 0], reference[:, 1], n, total_e, is_already_csls=is_csls),
        }
        e5 = wuxi_audit[m_name]['eval_5483']
        print(
            f"WUXI {m_name:12s} | eval_5483 Row-CSLS5 (No Sinkhorn) MRR={e5['rowwise_csls5_nosinkhorn']['mrr']*100:.2f}% H@1={e5['rowwise_csls5_nosinkhorn']['hits1']*100:.2f}% H@10={e5['rowwise_csls5_nosinkhorn']['hits10']*100:.2f}% "
            f"| Sinkhorn200 MRR={e5['csls5_sinkhorn200']['mrr']*100:.2f}%"
        )
    del sim_lightea, sim_lighttea, sim_htea, sim_rctea, sim_mgtea, msea_comp, cf, cs, cc, csls_neural_tri, csls_topo, csls_idf, csls_coupled_prop
    gc.collect()
    torch.cuda.empty_cache()
    return wuxi_audit


def run_dicews_relation_and_nosinkhorn_audit():
    """Compare Shared Relation Labels (242 shared) vs. Strictly Disjoint Relations (0 shared) on DICEWS-1K, plus No-Sinkhorn open-pool scores."""
    raw_dir = ROOT_EXP / 'datasets/ICEWS05-15'
    cache = ROOT_EXP / 'runs/dicews/cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m

    source_ids = np.asarray(sorted(read_vocab(raw_dir / 'ent_ids_1')), dtype=np.int64)
    target_ids = np.asarray(sorted(read_vocab(raw_dir / 'ent_ids_2')), dtype=np.int64)
    source_index = {int(v): i for i, v in enumerate(source_ids)}
    target_index = {int(v): i + n for i, v in enumerate(target_ids)}

    with np.load(cache / 'splits.npz') as splits:
        train_pair, test_pair = splits['train'], splits['test']
    reference_pairs = load_pairs(raw_dir / 'ref_pairs', source_index, target_index)

    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    rel_names_shared = protocol['relation_names']
    ri_shared = {label: i for i, label in enumerate(rel_names_shared)}

    # Disjoint relation namespace: graph 1 uses 0..246, graph 2 uses 247..492
    rv0_keys = sorted(rv[0].keys())
    rv1_keys = sorted(rv[1].keys())
    ri_disjoint_0 = {k: idx for idx, k in enumerate(rv0_keys)}
    ri_disjoint_1 = {k: len(rv0_keys) + idx for idx, k in enumerate(rv1_keys)}
    total_r_shared = len(rel_names_shared)  # 251
    total_r_disjoint = len(rv0_keys) + len(rv1_keys)  # 493

    triples_shared, triples_disjoint = [], []
    all_times = set()
    for g in (0, 1):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, tau = map(int, line.split()[:4])
            h = source_index[h_raw] if g == 0 else target_index[h_raw]
            t = source_index[t_raw] if g == 0 else target_index[t_raw]
            r_sh = ri_shared[rv[g][r_raw]]
            r_dj = ri_disjoint_0[r_raw] if g == 0 else ri_disjoint_1[r_raw]
            all_times.add(tau)
            triples_shared.append((h, r_sh, t, tau))
            triples_disjoint.append((h, r_dj, t, tau))

    time_sorted = sorted(all_times)
    ti = {tau: idx + 1 for idx, tau in enumerate(time_sorted)}
    total_t = len(time_sorted)

    # Dice time similarity on DICEWS
    entity_time_sets = [set() for _ in range(total_e)]
    for h, _, t, tau in triples_shared:
        tid = ti[tau] - 1
        entity_time_sets[h].add(tid)
        entity_time_sets[t].add(tid)
    inc = torch.zeros(total_e, total_t, dtype=torch.float32, device='cuda')
    for eid, tset in enumerate(entity_time_sets):
        if tset:
            inc[eid, list(tset)] = 1.0
    inter = inc[:n] @ inc[n:].T
    denom = inc[:n].sum(1, keepdim=True) + inc[n:].sum(1)[None, :]
    S_time_dice = torch.where(denom > 0, 2.0 * inter / denom.clamp_min(1.0), torch.zeros_like(inter))

    def eval_lighttea_under_rel_mode(triples_list, num_r, time_similarity):
        ee_edges, er_edges, re_edges, et_edges, te_edges = Counter(), Counter(), Counter(), Counter(), Counter()
        for h, r, t, tau in triples_list:
            tid = ti[tau] - 1
            ee_edges[(h, t)] += 1.0
            ee_edges[(t, h)] += 1.0
            er_edges[(h, 2 * r)] += 1.0
            er_edges[(t, 2 * r + 1)] += 1.0
            re_edges[(2 * r, h)] += 1.0
            re_edges[(2 * r + 1, t)] += 1.0
            et_edges[(h, tid)] += 1.0
            et_edges[(t, tid)] += 1.0
            te_edges[(tid, h)] += 1.0
            te_edges[(tid, t)] += 1.0

        def make_sp(edge_counter, shape):
            rows, cols, vals = [], [], []
            row_sums = defaultdict(float)
            for (r_i, c_i), v in edge_counter.items():
                w = math.log1p(v)
                rows.append(r_i)
                cols.append(c_i)
                vals.append(w)
                row_sums[r_i] += w
            vals = [v / max(row_sums[r_i], 1e-12) for r_i, v in zip(rows, vals)]
            return torch.sparse_coo_tensor(torch.tensor([rows, cols], device='cuda'), torch.tensor(vals, dtype=torch.float32, device='cuda'), size=shape).coalesce()

        A_EE = make_sp(ee_edges, (total_e, total_e))
        A_ER = make_sp(er_edges, (total_e, 2 * num_r))
        A_RE = make_sp(re_edges, (2 * num_r, total_e))
        A_ET = make_sp(et_edges, (total_e, total_t))
        A_TE = make_sp(te_edges, (total_t, total_e))

        gen = torch.Generator(device='cuda').manual_seed(2026)
        seed_vecs = F.normalize(torch.randn(len(train_pair), 300, generator=gen, device='cuda'), dim=-1)
        L_e0 = torch.zeros(total_e, 300, device='cuda')
        for idx_s, (u, v) in enumerate(train_pair):
            L_e0[int(u)] = seed_vecs[idx_s]
            L_e0[int(v)] = seed_vecs[idx_s]

        L_e_rel, L_r, curr_e = [L_e0], torch.zeros(2 * num_r, 300, device='cuda'), L_e0
        for _ in range(2):
            curr_e, L_r = F.normalize(torch.sparse.mm(A_EE, curr_e) + torch.sparse.mm(A_ER, L_r), dim=-1), F.normalize(torch.sparse.mm(A_RE, curr_e), dim=-1)
            L_e_rel.append(curr_e)
        H_rel = F.normalize(torch.cat(L_e_rel, dim=-1), dim=-1)

        L_e_time, L_t, curr_et = [L_e0], torch.zeros(total_t, 300, device='cuda'), L_e0
        for _ in range(2):
            curr_et, L_t = F.normalize(torch.sparse.mm(A_EE, curr_et) + torch.sparse.mm(A_ET, L_t), dim=-1), F.normalize(torch.sparse.mm(A_TE, curr_et), dim=-1)
            L_e_time.append(curr_et)
        H_time = F.normalize(torch.cat(L_e_time, dim=-1), dim=-1)

        sim_lea = H_rel[:n] @ H_rel[n:].T
        H_ltea = F.normalize(0.7 * H_rel + 0.3 * H_time, dim=-1)
        sim_ltea = 0.7 * (H_ltea[:n] @ H_ltea[n:].T) + 0.3 * time_similarity
        return (
            score_open_pool_all_modes(sim_lea, test_pair, np.arange(n), np.arange(n, total_e), n, total_e),
            score_open_pool_all_modes(sim_ltea, test_pair, np.arange(n), np.arange(n, total_e), n, total_e),
        )

    lea_shared, ltea_shared = eval_lighttea_under_rel_mode(triples_shared, total_r_shared, S_time_dice)
    lea_disjoint, ltea_disjoint = eval_lighttea_under_rel_mode(triples_disjoint, total_r_disjoint, S_time_dice)

    # Also evaluate our frozen MSEA-Align (shared relation induced, step 325) under No-Sinkhorn vs Sinkhorn
    dicews_run = ROOT_EXP / 'runs/dicews/shared'
    z_msea_shared = torch.from_numpy(np.load(dicews_run / 'best_vectors.npy')).cuda()
    sim_msea_shared = z_msea_shared[:n] @ z_msea_shared[n:].T
    msea_shared_scores = score_open_pool_all_modes(sim_msea_shared, test_pair, np.arange(n), np.arange(n, total_e), n, total_e)
    msea_shared_closed = score_open_pool_all_modes(sim_msea_shared, test_pair, reference_pairs[:, 0], reference_pairs[:, 1], n, total_e)
    del sim_msea_shared, z_msea_shared, S_time_dice, inc, inter, denom
    gc.collect()
    torch.cuda.empty_cache()

    best_z_dj = torch.from_numpy(np.load(ROOT_EXP / 'runs/dicews/disjoint/best_vectors.npy'))
    sim_msea_dj = best_z_dj[:n].cuda().float() @ best_z_dj[n:].cuda().float().T
    msea_dj_scores = score_open_pool_all_modes(sim_msea_dj, test_pair, np.arange(n), np.arange(n, total_e), n, total_e)
    msea_dj_closed = score_open_pool_all_modes(sim_msea_dj, test_pair, reference_pairs[:, 0], reference_pairs[:, 1], n, total_e)
    del sim_msea_dj
    torch.cuda.empty_cache()

    # Also get No-Sinkhorn (Row-CSLS5 & Cosine) for RCTEA on DICEWS open pool
    rctea_raw = np.load(ROOT_EXP / 'runs/dicews/rctea/embeddings.npy')
    raw_order = np.concatenate([source_ids, target_ids])
    rctea_vecs = torch.from_numpy(rctea_raw[raw_order]).cuda().float()
    rctea_vecs = rctea_vecs / (rctea_vecs.norm(dim=-1, keepdim=True) + 1e-5)
    sim_rctea = rctea_vecs[:n] @ rctea_vecs[n:].T
    rctea_disjoint_open = score_open_pool_all_modes(sim_rctea, test_pair, np.arange(n), np.arange(n, total_e), n, total_e)
    rctea_disjoint_closed = score_open_pool_all_modes(sim_rctea, test_pair, reference_pairs[:, 0], reference_pairs[:, 1], n, total_e)

    return dict(
        LightEA_shared_rel_open=lea_shared,
        LightEA_disjoint_rel_open=lea_disjoint,
        LightTEA_shared_rel_open=ltea_shared,
        LightTEA_disjoint_rel_open=ltea_disjoint,
        RCTEA_disjoint_rel_open=rctea_disjoint_open,
        RCTEA_disjoint_rel_closed=rctea_disjoint_closed,
        MSEA_Align_shared_rel_open=msea_shared_scores,
        MSEA_Align_shared_rel_closed=msea_shared_closed,
        MSEA_Align_disjoint_rel_open=msea_dj_scores,
        MSEA_Align_disjoint_rel_closed=msea_dj_closed,
    )


def main():
    suite_dir = ROOT_EXP / 'experiments/wuxi_suite_20261108'
    wuxi_audit = eval_wuxi_nosinkhorn_audit()
    dicews_audit = run_dicews_relation_and_nosinkhorn_audit()
    payload = dict(
        state='completed_protocol_audits',
        wuxi_open_pool_nosinkhorn_vs_sinkhorn=wuxi_audit,
        dicews_shared_vs_disjoint_relations_and_nosinkhorn=dicews_audit,
    )
    save_json(suite_dir / 'protocol_audits_nosink_and_disjoint_rel.json', payload)
    print('ALL PROTOCOL AUDITS COMPLETED!', flush=True)


if __name__ == '__main__':
    main()
