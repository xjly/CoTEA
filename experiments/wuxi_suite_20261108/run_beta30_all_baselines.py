"""Run LightEA, STEA, LightTEA, MGTEA, HTEA, and RCTEA on BETA (30% seeds) under unified protocol."""
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F

ROOT_EXP = Path(__file__).resolve().parents[2]
RCTEA_ROOT = ROOT_EXP / 'third_party/rctea/dicews'
MGTEA_SRC = ROOT_EXP / 'third_party/mgtea'
sys.path.insert(0, str(MGTEA_SRC))
sys.path.insert(0, str(RCTEA_ROOT))

from reproduction.benchmark_retrieval import balance_inplace
from reproduction.data import read_vocab, save_json
from reproduction.priority_hybrid_train import validation_groups, interval_time_groups
from reproduction.train import average_ranks


@torch.inference_mode()
def evaluate_beta30_closed(sim_28255, groups, block=256):
    """Evaluate a (28255, 28255) float32 similarity matrix on GPU under native_sinkhorn10 and csls5_sinkhorn200."""
    nr = len(sim_28255)
    device = sim_28255.device

    def m_fn(arr):
        return dict(
            mrr=float((1.0 / arr).mean()),
            hits1=float((arr <= 1.0).mean()),
            hits10=float((arr <= 10.0).mean()),
            queries=int(len(arr)),
        )

    def summarize_ranks(r):
        return dict(
            overall=m_fn(r),
            groups={k: m_fn(r[mask]) for k, mask in groups.items() if mask.any()},
        )

    # 1. Native Sinkhorn10 (exp(50*sim), 10 passes in float32)
    work = sim_28255.float().clone()
    work.mul_(50.0).exp_()
    for _ in range(10):
        work.div_(work.sum(dim=1, keepdim=True) + 1e-6)
        work.div_(work.sum(dim=0, keepdim=True) + 1e-6)
    ranks_nat = []
    for s in range(0, nr, 128):
        e = min(nr, s + 128)
        gold = torch.arange(s, e, device=device)
        ranks_nat.append(average_ranks(work[s:e], gold).cpu().numpy())
    ranks_nat = np.concatenate(ranks_nat).astype(np.float64)
    del work
    torch.cuda.empty_cache()

    # 2. CSLS-5 + 200-round Sinkhorn (T=0.05 in float64)
    rx = torch.empty(nr, device=device, dtype=torch.float64)
    ry = torch.empty(nr, device=device, dtype=torch.float64)
    for s in range(0, nr, block):
        e = min(nr, s + block)
        rx[s:e] = sim_28255[s:e].topk(5, dim=1).values.double().mean(1)
        ry[s:e] = sim_28255[:, s:e].topk(5, dim=0).values.double().mean(0)

    csls_mat = torch.empty((nr, nr), device=device, dtype=torch.float64)
    for s in range(0, nr, block):
        e = min(nr, s + block)
        csls_mat[s:e] = 2.0 * sim_28255[s:e].double() - rx[s:e, None] - ry[None, :]
    u, v = balance_inplace(csls_mat, temperature=0.05, iterations=200, block=block)
    ranks_csls_sink = []
    for s in range(0, nr, 128):
        e = min(nr, s + 128)
        score = csls_mat[s:e] + u[s:e, None] + v[None, :]
        gold = torch.arange(s, e, device=device)
        ranks_csls_sink.append(average_ranks(score, gold).cpu().numpy())
    ranks_csls_sink = np.concatenate(ranks_csls_sink).astype(np.float64)
    del csls_mat, rx, ry, u, v
    gc.collect()
    torch.cuda.empty_cache()

    return dict(
        native_sinkhorn10=summarize_ranks(ranks_nat),
        csls5_sinkhorn200=summarize_ranks(ranks_csls_sink),
    )


def load_beta30_context():
    raw_dir = ROOT_EXP / 'datasets/BETA'
    cache = ROOT_EXP / 'runs/beta30/cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m
    assert (n, m) == (42666, 42297)

    with np.load(cache / 'splits.npz') as splits:
        train_pair = splits['train'].copy()
        ref_pair = splits['reference'].copy()
    assert len(train_pair) == 12109 and len(ref_pair) == 28255

    events = np.load(cache / 'events.npz')['events']
    groups = validation_groups(cache, train_pair, ref_pair)
    groups.update(interval_time_groups(events, ref_pair))

    ev = [read_vocab(raw_dir / f'ent_ids_{i}') for i in (1, 2)]
    raw_ids = [sorted(v) for v in ev]
    indices = [{v: j + off for j, v in enumerate(ids)} for ids, off in zip(raw_ids, (0, n))]
    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    rel_names = protocol['relation_names']
    ri = {label: i for i, label in enumerate(rel_names)}

    return raw_dir, cache, protocol, n, m, total_e, train_pair, ref_pair, groups, indices, rv, ri


def compute_beta_dice_time_sim(raw_dir, indices, ref_pair, total_e):
    """Compute Dice time similarity on the 28255x28255 reference pairs using endpoint timestamps."""
    t_vocab = {}
    ent_times = [set() for _ in range(total_e)]
    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, ts, te = line.split()
            h = indices[g][int(h_raw)]
            t = indices[g][int(t_raw)]
            for lbl in (ts, te):
                if lbl != '~':
                    if lbl not in t_vocab:
                        t_vocab[lbl] = len(t_vocab)
                    tid = t_vocab[lbl]
                    ent_times[h].add(tid)
                    ent_times[t].add(tid)

    num_t = len(t_vocab)
    rows_l, cols_l = [], []
    rows_r, cols_r = [], []
    cnt_l = np.zeros(len(ref_pair), dtype=np.float32)
    cnt_r = np.zeros(len(ref_pair), dtype=np.float32)
    for i, (u, v) in enumerate(ref_pair):
        tu = ent_times[int(u)]
        tv = ent_times[int(v)]
        cnt_l[i] = len(tu)
        cnt_r[i] = len(tv)
        for tid in tu:
            rows_l.append(i)
            cols_l.append(tid)
        for tid in tv:
            rows_r.append(i)
            cols_r.append(tid)

    sp_l = torch.sparse_coo_tensor(
        torch.tensor([rows_l, cols_l], dtype=torch.long, device='cuda'),
        torch.ones(len(rows_l), dtype=torch.float32, device='cuda'),
        size=(len(ref_pair), num_t),
    ).coalesce()
    sp_r = torch.sparse_coo_tensor(
        torch.tensor([rows_r, cols_r], dtype=torch.long, device='cuda'),
        torch.ones(len(rows_r), dtype=torch.float32, device='cuda'),
        size=(len(ref_pair), num_t),
    ).coalesce()
    # Compute inter = sp_l @ sp_r.T in blocks of 2048 rows to save memory
    sim_dice = torch.empty((len(ref_pair), len(ref_pair)), dtype=torch.float32, device='cuda')
    sp_r_t = sp_r.transpose(0, 1).coalesce()
    cl_t = torch.tensor(cnt_l, device='cuda')
    cr_t = torch.tensor(cnt_r, device='cuda')
    for s in range(0, len(ref_pair), 1024):
        e = min(len(ref_pair), s + 1024)
        # Slice rows s:e
        mask = (sp_l.indices()[0] >= s) & (sp_l.indices()[0] < e)
        sub_idx = sp_l.indices()[:, mask].clone()
        sub_idx[0] -= s
        sub_l = torch.sparse_coo_tensor(sub_idx, sp_l.values()[mask], size=(e - s, num_t), device='cuda')
        inter = torch.sparse.mm(sub_l, sp_r_t).to_dense()
        denom = cl_t[s:e, None] + cr_t[None, :]
        sim_dice[s:e] = torch.where(denom > 0, 2.0 * inter / denom.clamp_min(1.0), torch.zeros_like(inter))
    del sp_l, sp_r, sp_r_t
    torch.cuda.empty_cache()
    return sim_dice, t_vocab


def task_light(out_dir):
    raw_dir, cache, protocol, n, m, total_e, train_pair, ref_pair, groups, indices, rv, ri = load_beta30_context()
    total_r = len(ri)
    t0 = time.perf_counter()
    sim_dice, t_vocab = compute_beta_dice_time_sim(raw_dir, indices, ref_pair, total_e)
    total_t = len(t_vocab)

    ee_edges, er_edges, re_edges, et_edges, te_edges = Counter(), Counter(), Counter(), Counter(), Counter()
    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, ts, te = line.split()
            h = indices[g][int(h_raw)]
            t = indices[g][int(t_raw)]
            r = ri[rv[g][int(r_raw)]]
            ee_edges[(h, t)] += 1.0
            ee_edges[(t, h)] += 1.0
            er_edges[(h, 2 * r)] += 1.0
            er_edges[(t, 2 * r + 1)] += 1.0
            re_edges[(2 * r, h)] += 1.0
            re_edges[(2 * r + 1, t)] += 1.0
            for lbl in (ts, te):
                if lbl != '~':
                    tid = t_vocab[lbl]
                    et_edges[(h, tid)] += 1.0
                    et_edges[(t, tid)] += 1.0
                    te_edges[(tid, h)] += 1.0
                    te_edges[(tid, t)] += 1.0

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
    d = 300
    seed_vecs = F.normalize(torch.randn(len(train_pair), d, generator=gen, device='cuda'), dim=-1)
    L_e0 = torch.zeros(total_e, d, device='cuda')
    for idx_s, (u, v) in enumerate(train_pair):
        L_e0[int(u)] = seed_vecs[idx_s]
        L_e0[int(v)] = seed_vecs[idx_s]

    L_e_rel = [L_e0]
    L_r = torch.zeros(2 * total_r, d, device='cuda')
    curr_e = L_e0
    for _ in range(2):
        next_e = F.normalize(torch.sparse.mm(A_EE, curr_e) + torch.sparse.mm(A_ER, L_r), dim=-1)
        L_r = F.normalize(torch.sparse.mm(A_RE, curr_e), dim=-1)
        curr_e = next_e
        L_e_rel.append(curr_e)
    H_rel = F.normalize(torch.cat(L_e_rel, dim=-1), dim=-1)

    L_e_time = [L_e0]
    L_t = torch.zeros(total_t, d, device='cuda')
    curr_et = L_e0
    for _ in range(2):
        next_et = F.normalize(torch.sparse.mm(A_EE, curr_et) + torch.sparse.mm(A_ET, L_t), dim=-1)
        L_t = F.normalize(torch.sparse.mm(A_TE, curr_et), dim=-1)
        curr_et = next_et
        L_e_time.append(curr_et)
    H_time = F.normalize(torch.cat(L_e_time, dim=-1), dim=-1)
    del A_EE, A_ER, A_RE, A_ET, A_TE
    torch.cuda.empty_cache()

    ref_l = torch.as_tensor(ref_pair[:, 0], device='cuda')
    ref_r = torch.as_tensor(ref_pair[:, 1], device='cuda')

    sim_lea = H_rel[ref_l] @ H_rel[ref_r].T
    res_lea = evaluate_beta30_closed(sim_lea, groups)
    del sim_lea
    torch.cuda.empty_cache()

    H_ltea = F.normalize(0.7 * H_rel + 0.3 * H_time, dim=-1)
    sim_ltea = 0.7 * (H_ltea[ref_l] @ H_ltea[ref_r].T) + 0.3 * sim_dice
    res_ltea = evaluate_beta30_closed(sim_ltea, groups)
    res_ltea['elapsed_seconds'] = time.perf_counter() - t0

    save_json(out_dir / 'beta30_light.json', {'LightEA': res_lea, 'LightTEA': res_ltea})
    print('BETA30 LightEA:', res_lea['csls5_sinkhorn200']['overall'], flush=True)
    print('BETA30 LightTEA:', res_ltea['csls5_sinkhorn200']['overall'], flush=True)


def build_beta30_int_matrices(raw_dir, indices, rv, ri, total_e, total_r):
    """Build RCTEA/HTEA matrices from BETA triples_1c / triples_2c (integer year endpoints)."""
    y_vocab = {0: 0}
    triples1, triples2 = [], []
    time1, time2 = set(), set()
    time_int1, time_int2 = set(), set()

    for g, t_list, t_set, ti_set in [(0, triples1, time1, time_int1), (1, triples2, time2, time_int2)]:
        for line in (ROOT_EXP / 'third_party/rctea/beta' / f'triples_{g + 1}c').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_raw, r_raw, t_raw, ts_raw, te_raw = map(int, line.split())
            h = indices[g][h_raw]
            t = indices[g][t_raw]
            r = ri[rv[g][r_raw]]
            for yv in (ts_raw, te_raw):
                if yv not in y_vocab:
                    y_vocab[yv] = len(y_vocab)
            ts = y_vocab[ts_raw]
            te = y_vocab[te_raw]
            if ts != 0 and te == 0:
                te = ts
            if te != 0 and ts == 0:
                ts = te
            t_int = (ts, te)
            t_set.add(ts)
            t_set.add(te)
            ti_set.add(t_int)
            t_list.append([h, r, t, ts, te])

    all_triples = triples1 + triples2
    tem_valid_set = time1 & time2
    tem_int_valid_set = time_int1 & time_int2
    all_time_int = sorted(time_int1 | time_int2)
    if (0, 0) not in all_time_int:
        all_time_int.insert(0, (0, 0))
    time_int_dict_i = {ti_k: i for i, ti_k in enumerate(all_time_int)}
    zero_index = time_int_dict_i[(0, 0)]

    rec_tem_count = defaultdict(Counter)
    rec_tem_int_count = defaultdict(Counter)
    for h, r, t, ts, te in all_triples:
        for e in (h, t):
            rec_tem_count[e][ts] += 1
            rec_tem_count[e][te] += 1
            rec_tem_int_count[e][(ts, te)] += 1

    ent_size = total_e + 1
    rel_size = total_r + 1
    time_size = len(y_vocab) + 1
    time_int_size = len(all_time_int)

    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    for i in range(ent_size):
        adj_features[i, i] = 1
    rel_in = np.zeros((ent_size, rel_size), dtype=np.float32)
    rel_out = np.zeros((ent_size, rel_size), dtype=np.float32)
    time_link = np.zeros((ent_size, time_size), dtype=np.float32)
    time_int_link = np.zeros((ent_size, time_int_size), dtype=np.float32)

    for i in range(total_e):
        if i not in rec_tem_count:
            time_link[i, 0] = 1.0
        else:
            for j, cnt in rec_tem_count[i].items():
                if j in tem_valid_set:
                    time_link[i, j] = np.log1p(cnt)
        if i not in rec_tem_int_count:
            time_int_link[i, zero_index] = 1.0
        else:
            for j, cnt in rec_tem_int_count[i].items():
                if j in tem_int_valid_set:
                    time_int_link[i, time_int_dict_i[j]] = np.log1p(cnt)

    for h, r, t, ts, te in all_triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] += 1
        adj_features[t, h] += 1
        adj_features[h, -1] += 1
        adj_features[t, -1] += 1
        rel_in[h, r] += 1
        rel_out[t, r] += 1
        rel_in[h, total_r] += 1
        rel_out[t, total_r] += 1
    adj_matrix[total_e, total_e] = 1

    time_features = sp.coo_matrix(time_link / (np.sum(time_link, axis=-1, keepdims=True) + 0.01))
    time_int_features = sp.coo_matrix(time_int_link / (np.sum(time_int_link, axis=-1, keepdims=True) + 0.01))
    rel_concat = np.log1p(np.concatenate([rel_in, rel_out], axis=1))
    rel_features = sp.coo_matrix(rel_concat / (np.sum(rel_concat, axis=-1, keepdims=True) + 0.01))

    adj_coo = adj_features.tocoo()
    row_sums = np.asarray(sp.coo_matrix((np.log1p(adj_coo.data), (adj_coo.row, adj_coo.col)), shape=adj_coo.shape).sum(axis=1)).squeeze() + 0.01
    norm_vals = np.log1p(adj_coo.data) / row_sums[adj_coo.row]
    adj_features = sp.coo_matrix((norm_vals, (adj_coo.row, adj_coo.col)), shape=adj_coo.shape)

    return (
        adj_matrix, adj_features, rel_features, time_features, time_int_features,
        time_int_dict_i, triples1, triples2,
    )


def task_stea_htea(out_dir):
    import model_o
    import model_r

    raw_dir, cache, protocol, n, m, total_e, train_pair, ref_pair, groups, indices, rv, ri = load_beta30_context()
    total_r = len(ri)
    (
        adj_matrix, adj_features, rel_features, time_features, time_int_features,
        time_int_dict_i, triples1, triples2,
    ) = build_beta30_int_matrices(raw_dir, indices, rv, ri, total_e, total_r)

    adj_idx = np.stack(adj_matrix.nonzero(), axis=1)
    rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
    time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
    time_int_matrix, time_int_val = np.stack(time_int_features.nonzero(), axis=1), time_int_features.data
    ent_size = adj_features.shape[0]
    zero_index = time_int_dict_i[(0, 0)]

    class DummyCfg:
        tau = 0.05
        ab_weight = 0.5
        replay = False
        neg_cross_kg = False

    device = 'cuda:0'
    ref_l = torch.as_tensor(ref_pair[:, 0], device=device)
    ref_r = torch.as_tensor(ref_pair[:, 1], device=device)

    # STEA
    t0 = time.perf_counter()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    m_rel = model_r.OverAll(
        node_size=ent_size, node_hidden=100,
        rel_size=rel_features.shape[1], rel_hidden=100,
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val, args=DummyCfg(),
        triple_size=len(adj_idx), dropout_rate=0.3,
        depth=2, device=device,
    ).to(device)
    opt_r = torch.optim.RMSprop(m_rel.parameters(), lr=0.005, weight_decay=0)
    train_shuf = train_pair.copy()
    for _ in range(20):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * 1024:(i + 1) * 1024] for i in range(len(train_shuf) // 1024 + 1)]:
            if not len(pairs):
                continue
            _, loss_e = m_rel([adj_idx, pairs])
            loss_e.backward(retain_graph=True)
            opt_r.step()
            opt_r.zero_grad()
    m_rel.eval()
    m_rel.return_loss = False
    with torch.no_grad():
        z_stea = F.normalize(m_rel([adj_idx, train_pair])[:total_e], dim=-1)
    del m_rel, opt_r
    torch.cuda.empty_cache()

    sim_dice, _ = compute_beta_dice_time_sim(raw_dir, indices, ref_pair, total_e)
    sim_stea = 0.7 * (z_stea[ref_l] @ z_stea[ref_r].T) + 0.3 * sim_dice
    del z_stea, sim_dice
    torch.cuda.empty_cache()
    res_stea = evaluate_beta30_closed(sim_stea, groups)
    res_stea['elapsed_seconds'] = time.perf_counter() - t0
    del sim_stea
    torch.cuda.empty_cache()

    # HTEA
    t1 = time.perf_counter()
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    m_htea = model_o.OverAll(
        node_size=ent_size, node_hidden=100, time_hidden=50,
        rel_size=rel_features.shape[1], rel_hidden=100,
        time_size=time_features.shape[1], time_int_size=time_int_features.shape[1],
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val,
        time_matrix=time_matrix, time_val=time_val, out_dim=4,
        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=DummyCfg(),
        triple_size=len(adj_idx), dropout_rate=0.3,
        depth=2, device=device,
    ).to(device)
    m_htea.zero_interval = zero_index
    m_htea.return_loss = False
    opt_h = torch.optim.RMSprop(m_htea.parameters(), lr=0.005, weight_decay=0)

    def htea_loss(align_input, embedding):
        left = torch.as_tensor(align_input[:, 0], dtype=torch.long, device=device)
        right = torch.as_tensor(align_input[:, 1], dtype=torch.long, device=device)
        l_emb, r_emb = embedding[left], embedding[right]
        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        emb_sq = torch.sum(torch.square(embedding), dim=1, keepdim=True).T
        r_neg_dis = torch.sum(torch.square(r_emb), dim=1, keepdim=True) + emb_sq - 2.0 * (r_emb @ embedding.T)
        l_neg_dis = torch.sum(torch.square(l_emb), dim=1, keepdim=True) + emb_sq - 2.0 * (l_emb @ embedding.T)
        mask = 1.0 - F.one_hot(left, num_classes=ent_size).float() - F.one_hot(right, num_classes=ent_size).float()
        l_loss = (pos_dis - l_neg_dis + 1.0) * mask
        r_loss = (pos_dis - r_neg_dis + 1.0) * mask
        with torch.no_grad():
            r_loss.data = (r_loss.data - r_loss.mean(-1, keepdim=True)) / r_loss.std(-1, keepdim=True).clamp_min(1e-6)
            l_loss.data = (l_loss.data - l_loss.mean(-1, keepdim=True)) / l_loss.std(-1, keepdim=True).clamp_min(1e-6)
        return torch.mean(torch.logsumexp(30.0 * l_loss + 10.0, dim=-1) + torch.logsumexp(30.0 * r_loss + 10.0, dim=-1))

    for _ in range(15):
        m_htea.train()
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * 1024:(i + 1) * 1024] for i in range(len(train_shuf) // 1024 + 1)]:
            if not len(pairs):
                continue
            out = m_htea([adj_idx, pairs])
            loss = htea_loss(pairs, out)
            opt_h.zero_grad()
            loss.backward()
            opt_h.step()
    m_htea.eval()
    with torch.no_grad():
        feat_h = m_htea([adj_idx, train_pair])[:total_e]
        feat_h = feat_h / (feat_h.norm(dim=-1, keepdim=True) + 1e-5)
        sim_htea = feat_h[ref_l] @ feat_h[ref_r].T
    del m_htea, opt_h, feat_h
    torch.cuda.empty_cache()
    res_htea = evaluate_beta30_closed(sim_htea, groups)
    res_htea['elapsed_seconds'] = time.perf_counter() - t1

    save_json(out_dir / 'beta30_stea_htea.json', {'STEA': res_stea, 'HTEA': res_htea})
    print('BETA30 STEA:', res_stea['csls5_sinkhorn200']['overall'], flush=True)
    print('BETA30 HTEA:', res_htea['csls5_sinkhorn200']['overall'], flush=True)


def task_mgtea(out_dir):
    from MGTEA_models import OverAll as MGTEAOverAll
    raw_dir, cache, protocol, n, m, total_e, train_pair, ref_pair, groups, indices, rv, ri = load_beta30_context()
    total_r = len(ri)
    t0 = time.perf_counter()

    year_vocab = {'~': 0}
    fine_vocab = {'~': 0}
    raw_triples = []
    for g in range(2):
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_s, r_s, t_s, ts, te = line.split()
            h = indices[g][int(h_s)]
            t = indices[g][int(t_s)]
            r = ri[rv[g][int(r_s)]] + 1
            for lbl in (ts, te):
                y_id, f_id = 0, 0
                if lbl != '~' and '.' in lbl:
                    y_part, md_part = lbl.rsplit('.', 1)
                    if y_part not in year_vocab:
                        year_vocab[y_part] = len(year_vocab)
                    y_id = year_vocab[y_part]
                    if md_part not in ('0101', '0100', '0000'):
                        if lbl not in fine_vocab:
                            fine_vocab[lbl] = len(fine_vocab)
                        f_id = fine_vocab[lbl]
                raw_triples.append((h, r, t, y_id, f_id))

    ent_size = total_e
    rel_size = 2 * (total_r + 2)
    time_size = len(year_vocab) + 1
    time_sizeT = len(fine_vocab) + 1

    ent_mat = sp.lil_matrix((ent_size, ent_size))
    for i in range(ent_size):
        ent_mat[i, i] = 1.0
    rel_in = np.zeros((ent_size, total_r + 2), dtype=np.float32)
    rel_out = np.zeros((ent_size, total_r + 2), dtype=np.float32)
    time_link_y = defaultdict(Counter)
    time_link_d = defaultdict(Counter)

    radj = []
    for h, r, t, y_id, f_id in raw_triples:
        ent_mat[h, t] = 1
        ent_mat[t, h] = 1
        radj.append((h, t, r, y_id, f_id))
        radj.append((t, h, r + total_r + 2, y_id, f_id))
        rel_out[h, r] += 1
        rel_in[t, r] += 1
        time_link_y[h][y_id] += 1
        time_link_y[t][y_id] += 1
        time_link_d[h][f_id] += 1
        time_link_d[t][f_id] += 1

    count = -1
    seen = set()
    deg_pair = {}
    r_index, t_index, t_indexT, r_val, adj_pairs = [], [], [], [], []
    for h, t, r_m, y_id, f_id in sorted(radj, key=lambda x: (x[0], x[1])):
        key = (h, t)
        if key not in seen:
            count += 1
            seen.add(key)
            deg_pair[count] = 0
            adj_pairs.append([h, t])
        deg_pair[count] += 1
        r_index.append([count, r_m])
        t_index.append([count, y_id])
        t_indexT.append([count, f_id])
        r_val.append(1.0)
    for i in range(len(r_val)):
        r_val[i] /= deg_pair[r_index[i][0]]

    adj_index = np.array(adj_pairs, dtype=np.int64)
    r_index = np.array(r_index, dtype=np.int64)
    t_index = np.array(t_index, dtype=np.int64)
    t_indexT = np.array(t_indexT, dtype=np.int64)
    r_val = np.array(r_val, dtype=np.float32)

    ent_matrix = np.stack(ent_mat.tocoo().nonzero(), axis=1)
    rel_matrix = np.stack(sp.coo_matrix(np.concatenate([rel_in, rel_out], axis=1)).nonzero(), axis=1)
    time_matrix = np.array([[e, k] for e, c in time_link_y.items() for k in c], dtype=np.int64)
    time_matrixT = np.array([[e, k] for e, c in time_link_d.items() for k in c], dtype=np.int64)

    device = 'cuda:0'
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    model = MGTEAOverAll(
        node_size=ent_size, node_hidden=100,
        rel_size=rel_size, rel_hidden=100,
        time_size=time_size, time_sizeT=time_sizeT,
        triple_size=len(adj_index),
        rel_matrix=rel_matrix, ent_matrix=ent_matrix,
        time_matrix=time_matrix, time_matrixT=time_matrixT,
        dropout_rate=0.3, depth=2, device=device,
    ).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=0.005, weight_decay=0)
    inputs = [adj_index, r_index, r_val, t_index, rel_matrix, ent_matrix, t_indexT]

    def mgtea_loss(align_input, embedding):
        left = torch.as_tensor(align_input[:, 0], dtype=torch.long, device=device)
        right = torch.as_tensor(align_input[:, 1], dtype=torch.long, device=device)
        l_emb, r_emb = embedding[left], embedding[right]
        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        emb_sq = torch.sum(torch.square(embedding), dim=1, keepdim=True).T
        r_neg_dis = torch.sum(torch.square(r_emb), dim=1, keepdim=True) + emb_sq - 2.0 * (r_emb @ embedding.T)
        l_neg_dis = torch.sum(torch.square(l_emb), dim=1, keepdim=True) + emb_sq - 2.0 * (l_emb @ embedding.T)
        mask = 1.0 - F.one_hot(left, num_classes=ent_size).float() - F.one_hot(right, num_classes=ent_size).float()
        l_loss = (pos_dis - l_neg_dis + 1.0) * mask
        r_loss = (pos_dis - r_neg_dis + 1.0) * mask
        return torch.mean(torch.logsumexp(30.0 * l_loss + 10.0, dim=-1) + torch.logsumexp(30.0 * r_loss + 10.0, dim=-1))

    train_shuf = train_pair.copy()
    for _ in range(15):
        model.train()
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * 1024:(i + 1) * 1024] for i in range(len(train_shuf) // 1024 + 1)]:
            if not len(pairs):
                continue
            out = model(inputs)
            loss = mgtea_loss(pairs, out)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    ref_l = torch.as_tensor(ref_pair[:, 0], device=device)
    ref_r = torch.as_tensor(ref_pair[:, 1], device=device)
    with torch.no_grad():
        emb = model(inputs)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-5)
        sim1 = emb[ref_l] @ emb[ref_r].T
    del model, opt, emb
    torch.cuda.empty_cache()

    # MGTEA 3-hop propagated coarse (year) and fine (day) time similarities
    dr = Counter(r for _, r, _, _, _ in raw_triples)
    total_tr = len(raw_triples)
    s_rows, s_cols, s_vals = [], [], []
    for i in range(ent_size):
        s_rows.append(i)
        s_cols.append(i)
        s_vals.append(math.log(max(total_tr / ent_size, 1.01)))
    for h, r, t, _, _ in raw_triples:
        w = math.log(max(total_tr / dr[r], 1.01))
        s_rows.append(h)
        s_cols.append(t)
        s_vals.append(w)
    A_rel = torch.sparse_coo_tensor(
        torch.tensor([s_rows, s_cols], dtype=torch.long, device=device),
        torch.tensor(s_vals, dtype=torch.float32, device=device),
        size=(ent_size, ent_size),
    ).coalesce()

    gen = torch.Generator(device=device).manual_seed(2026)

    def prop_projected_time_sim(t_dict, num_t, adjacency, proj_dim=512):
        rows, cols, vals = [], [], []
        for e, c in t_dict.items():
            for k, cnt in c.items():
                if k > 0:
                    rows.append(e)
                    cols.append(k)
                    vals.append(float(cnt))
        sp_t = torch.sparse_coo_tensor(
            torch.tensor([rows, cols], dtype=torch.long, device=device),
            torch.tensor(vals, dtype=torch.float32, device=device),
            size=(ent_size, num_t),
        ).coalesce()
        P = torch.randn(num_t, proj_dim, generator=gen, device=device) / math.sqrt(proj_dim)
        feat = F.normalize(torch.sparse.mm(sp_t, P), dim=-1)
        sim = feat[ref_l] @ feat[ref_r].T
        for _ in range(3):
            feat = F.normalize(torch.sparse.mm(adjacency, feat), dim=-1)
            sim = sim + (feat[ref_l] @ feat[ref_r].T)
        del sp_t, P, feat
        return sim / 4.0

    sim_year = prop_projected_time_sim(time_link_y, time_size, A_rel, proj_dim=512)
    sim_day = prop_projected_time_sim(time_link_d, time_sizeT, A_rel, proj_dim=512)
    sim4 = 0.85 * sim_year + 0.15 * sim_day
    del sim_year, sim_day, A_rel
    torch.cuda.empty_cache()

    sim_mgtea = (sim1 + 0.5 * sim4) / 1.5
    del sim1, sim4
    torch.cuda.empty_cache()

    res_mgtea = evaluate_beta30_closed(sim_mgtea, groups)
    res_mgtea['elapsed_seconds'] = time.perf_counter() - t0
    save_json(out_dir / 'beta30_mgtea.json', {'MGTEA': res_mgtea})
    print('BETA30 MGTEA:', res_mgtea['csls5_sinkhorn200']['overall'], flush=True)


def task_rctea(out_dir):
    import model_o
    import model_r
    import model_t
    from models import RelCNN
    from dgl_dual import DGMC

    raw_dir, cache, protocol, n, m, total_e, train_pair, ref_pair, groups, indices, rv, ri = load_beta30_context()
    total_r = len(ri)
    t0 = time.perf_counter()
    (
        adj_matrix, adj_features, rel_features, time_features, time_int_features,
        time_int_dict_i, triples1, triples2,
    ) = build_beta30_int_matrices(raw_dir, indices, rv, ri, total_e, total_r)

    adj_idx = np.stack(adj_matrix.nonzero(), axis=1)
    rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
    time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
    time_int_matrix, time_int_val = np.stack(time_int_features.nonzero(), axis=1), time_int_features.data
    node_size = adj_features.shape[0]
    rel_size = rel_features.shape[1]
    time_size = time_features.shape[1]
    time_int_size = time_int_features.shape[1]
    triple_size = len(adj_idx)
    zero_index = time_int_dict_i[(0, 0)]

    class DummyCfg:
        tau = 0.05
        ab_weight = 0.5
        replay = False
        neg_cross_kg = False

    cfgs = DummyCfg()
    device = 'cuda:0'
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    # BETA hyperparameters from RCTEA paper Section 5.1: rel=8, time=16, DGMC=110 (k=45, steps=25), over=1
    model_rel = model_r.OverAll(
        node_size=node_size, node_hidden=50,
        rel_size=rel_size, rel_hidden=50,
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val, args=cfgs,
        triple_size=triple_size, dropout_rate=0.3,
        depth=2, device=device,
    ).to(device)
    opt_r = torch.optim.RMSprop(model_rel.parameters(), lr=0.005, weight_decay=0)
    train_shuf = train_pair.copy()
    batch_size = 512
    for _ in range(8):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            _, loss_e = model_rel([adj_idx, pairs])
            loss_e.backward(retain_graph=True)
            opt_r.step()
            opt_r.zero_grad()

    model_time = model_t.OverAll(
        node_size=node_size, node_hidden=50, time_hidden=50,
        rel_size=rel_size, rel_hidden=50,
        time_size=time_size, time_int_size=time_int_size,
        time_matrix=time_matrix, time_val=time_val,
        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
        triple_size=triple_size, dropout_rate=0.3,
        depth=2, device=device,
    ).to(device)
    model_time.zero_interval = zero_index
    opt_t = torch.optim.RMSprop(model_time.parameters(), lr=0.005, weight_decay=0)
    for _ in range(16):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            _, loss_t = model_time([adj_idx, pairs])
            loss_t.backward(retain_graph=True)
            opt_t.step()
            opt_t.zero_grad()

    del opt_r, opt_t
    torch.cuda.empty_cache()

    all_pair = np.vstack((train_pair, ref_pair))
    indices_dic_1 = dict(zip(all_pair[:, 0], np.arange(0, len(all_pair), 1)))
    indices_dic_2 = dict(zip(all_pair[:, 1], np.arange(0, len(all_pair), 1)))

    def gen_edge_index(triples_list, d_map):
        rec = [[d_map[h], d_map[t]] for h, _, t, _, _ in triples_list if h in d_map and t in d_map]
        et = torch.LongTensor(rec)
        return torch.vstack((et[:, 0], et[:, 1])).to(device)

    edge_tensor_1 = gen_edge_index(triples1, indices_dic_1)
    edge_tensor_2 = gen_edge_index(triples2, indices_dic_2)
    train_pair_n = torch.LongTensor([[indices_dic_1[u], indices_dic_2[v]] for u, v in train_pair]).T.to(device)

    model_rel.set_att_gate(att=False, gate=True)
    model_rel.return_loss = False
    model_time.return_loss = False
    psi_3 = RelCNN(32, 32, 3, batch_norm=False, cat=True, lin=True, dropout=0.0)
    dgmc = DGMC(model_time, model_rel, psi_3, num_steps=0, k=45).to(device)
    opt_dgmc = torch.optim.Adam(dgmc.parameters(), lr=0.001)
    N_s = len(all_pair)
    inputs_full = [adj_idx, train_pair]

    for ep in range(1, 111):
        if ep == 31:
            dgmc.num_steps = 25
            dgmc.detach = True
        dgmc.train()
        select_index = np.arange(len(train_pair), N_s, 1)
        np.random.shuffle(select_index)
        selection_index = torch.hstack((train_pair_n[0, :], torch.LongTensor(select_index).to(device)))
        opt_dgmc.zero_grad()
        _, S_L = dgmc(
            inputs_full, indices_dic_1, indices_dic_2, None, edge_tensor_1,
            None, None, None, edge_tensor_2, None, None, selection_index, y=train_pair_n,
        )
        loss = dgmc.loss(S_L, train_pair_n)
        loss.backward()
        opt_dgmc.step()

    del dgmc, opt_dgmc, edge_tensor_1, edge_tensor_2
    torch.cuda.empty_cache()

    model_over = model_o.OverAll(
        node_size=node_size, node_hidden=50, time_hidden=50,
        rel_size=rel_size, rel_hidden=50,
        time_size=time_size, time_int_size=time_int_size,
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val,
        time_matrix=time_matrix, time_val=time_val, out_dim=4,
        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
        triple_size=triple_size, dropout_rate=0.3,
        depth=2, device=device,
    ).to(device)
    model_over.zero_interval = zero_index

    param_rel = model_rel.state_dict()
    param_time = model_time.state_dict()
    param_overall = model_over.state_dict()
    for k, v in param_rel.items():
        if k in param_overall and param_overall[k].shape == v.shape:
            param_overall[k] = v
    for k, v in param_time.items():
        if k in param_overall and k not in ('ent_emb', 'rel_emb') and param_overall[k].shape == v.shape:
            param_overall[k] = v
    model_over.load_state_dict(param_overall)
    model_over.set_att_gate(att=False, gate=True)

    opt_o = torch.optim.RMSprop(model_over.parameters(), lr=0.005, weight_decay=0)
    for _ in range(1):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            _, loss_o = model_over([adj_idx, pairs])
            loss_o.backward(retain_graph=True)
            opt_o.step()
            opt_o.zero_grad()

    model_over.eval()
    model_over.return_loss = False
    ref_l = torch.as_tensor(ref_pair[:, 0], device=device)
    ref_r = torch.as_tensor(ref_pair[:, 1], device=device)
    with torch.no_grad():
        feat = model_over([adj_idx, train_pair])[:total_e]
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-5)
        np.save(out_dir / 'beta30_rctea_embeddings.npy', feat.cpu().numpy())
        sim_rctea = feat[ref_l] @ feat[ref_r].T
    del model_over, model_rel, model_time, feat
    torch.cuda.empty_cache()

    res_rctea = evaluate_beta30_closed(sim_rctea, groups)
    res_rctea['elapsed_seconds'] = time.perf_counter() - t0
    save_json(out_dir / 'beta30_rctea.json', {'RCTEA': res_rctea})
    print('BETA30 RCTEA:', res_rctea['csls5_sinkhorn200']['overall'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['light', 'stea_htea', 'mgtea', 'rctea'], required=True)
    args = parser.parse_args()
    out_dir = ROOT_EXP / 'experiments/wuxi_suite_20261108/beta30_baselines'
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == 'light':
        task_light(out_dir)
    elif args.task == 'stea_htea':
        task_stea_htea(out_dir)
    elif args.task == 'mgtea':
        task_mgtea(out_dir)
    elif args.task == 'rctea':
        task_rctea(out_dir)


if __name__ == '__main__':
    main()
