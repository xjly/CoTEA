"""Run all 6 baselines (LightEA, STEA, LightTEA, MGTEA, HTEA, RCTEA) on DICEWS-1K under unified protocol."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys

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
from reproduction.data import read_vocab, load_pairs, save_json
from reproduction.priority_hybrid_train import validation_groups, common_day_mask
from reproduction.train import average_ranks


@torch.inference_mode()
def evaluate_dicews_sim(sim_n_by_m, test_pairs, reference_pairs, n, m, groups_test):
    """Evaluate an (n, m) similarity matrix on DICEWS-1K under full_9517x9537 and closed_7566x7566."""
    total_e = n + m

    def eval_pool(pool_rows, pool_cols, query_pairs, grp_dict, mode='csls5_sinkhorn200'):
        raw_sub = sim_n_by_m[pool_rows][:, pool_cols - n].float()
        if mode == 'native_sinkhorn10':
            sub = (50.0 * raw_sub).double().exp()
            for _ in range(10):
                sub.div_(sub.sum(dim=1, keepdim=True) + 1e-6)
                sub.div_(sub.sum(dim=0, keepdim=True) + 1e-6)
        else:
            rx = raw_sub.topk(5, dim=1).values.double().mean(1, keepdim=True)
            ry = raw_sub.topk(5, dim=0).values.double().mean(0, keepdim=True)
            sub = 2.0 * raw_sub.double() - rx - ry
            u, v = balance_inplace(sub, temperature=0.05, iterations=200, block=256)
            sub.add_(u[:, None]).add_(v[None, :])

        rowmap = np.full(total_e, -1, dtype=np.int64)
        rowmap[pool_rows] = np.arange(len(pool_rows))
        colmap = np.full(total_e, -1, dtype=np.int64)
        colmap[pool_cols] = np.arange(len(pool_cols))
        q = torch.as_tensor(rowmap[query_pairs[:, 0]], device='cuda')
        g = torch.as_tensor(colmap[query_pairs[:, 1]], device='cuda')
        r = average_ranks(sub[q], g).cpu().numpy()

        def m_fn(arr):
            return dict(
                mrr=float((1.0 / arr).mean()),
                hits1=float((arr <= 1.0).mean()),
                hits10=float((arr <= 10.0).mean()),
                queries=int(len(arr)),
            )
        res = dict(overall=m_fn(r))
        if grp_dict is not None:
            res['groups'] = {k: m_fn(r[mask]) for k, mask in grp_dict.items() if mask.any()}
        return res

    out = {}
    for mode in ('native_sinkhorn10', 'csls5_sinkhorn200'):
        out[mode] = dict(
            full_9517x9537_test6809=eval_pool(np.arange(n), np.arange(n, n + m), test_pairs, groups_test, mode),
            closed_7566x7566_test6809=eval_pool(reference_pairs[:, 0], reference_pairs[:, 1], test_pairs, groups_test, mode),
            closed_7566x7566_all7566=eval_pool(reference_pairs[:, 0], reference_pairs[:, 1], reference_pairs, None, mode),
        )
    return out


def main():
    raw_dir = ROOT_EXP / 'datasets/ICEWS05-15'
    cache = ROOT_EXP / 'runs/dicews/cache/priority'
    out_dir = ROOT_EXP / 'experiments/wuxi_suite_20261108/dicews_baselines'
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m
    assert (n, m) == (9517, 9537)

    source_ids = np.asarray(sorted(read_vocab(raw_dir / 'ent_ids_1')), dtype=np.int64)
    target_ids = np.asarray(sorted(read_vocab(raw_dir / 'ent_ids_2')), dtype=np.int64)
    source_index = {int(v): i for i, v in enumerate(source_ids)}
    target_index = {int(v): i + n for i, v in enumerate(target_ids)}
    raw_order = np.concatenate([source_ids, target_ids])

    with np.load(cache / 'splits.npz') as splits:
        train_pair, test_pair = splits['train'], splits['test']
    reference_pairs = load_pairs(raw_dir / 'ref_pairs', source_index, target_index)
    assert len(train_pair) == 1000 and len(test_pair) == 6809 and len(reference_pairs) == 7566

    events = np.load(cache / 'events.npz')['events']
    groups_test = validation_groups(cache, train_pair, test_pair)
    cd = common_day_mask(events, test_pair)
    groups_test.update(
        common_day=cd,
        no_common_day=~cd,
        sparse_common_day=groups_test['min_events_1or2'] & cd,
        sparse_no_common_day=groups_test['min_events_1or2'] & ~cd,
    )

    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    rel_names = protocol['relation_names']
    ri = {label: i for i, label in enumerate(rel_names)}
    total_r = len(rel_names)

    # Read raw triples mapped to our (0..n-1, n..n+m-1) entity order
    triples1, triples2 = [], []
    all_times = set()
    for g, t_list in [(0, triples1), (1, triples2)]:
        for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            parts = list(map(int, line.split()))
            h_raw, r_raw, t_raw, tau = parts[0], parts[1], parts[2], parts[3]
            h = source_index[h_raw] if g == 0 else target_index[h_raw]
            t = source_index[t_raw] if g == 0 else target_index[t_raw]
            r = ri[rv[g][r_raw]]
            all_times.add(tau)
            t_list.append((h, r, t, tau))

    time_sorted = sorted(all_times)
    ti = {tau: idx + 1 for idx, tau in enumerate(time_sorted)}
    total_t = len(time_sorted)

    results = {}

    # ---------------------------------------------------------
    # 1. RCTEA (Load existing verified s2026 embeddings)
    # ---------------------------------------------------------
    print('Evaluating RCTEA on DICEWS-1K...', flush=True)
    rctea_raw = np.load(ROOT_EXP / 'runs/dicews/rctea/embeddings.npy')
    rctea_vecs = torch.from_numpy(rctea_raw[raw_order]).cuda().float()
    rctea_vecs = rctea_vecs / (rctea_vecs.norm(dim=-1, keepdim=True) + 1e-5)
    sim_rctea = rctea_vecs[:n] @ rctea_vecs[n:].T
    results['RCTEA'] = evaluate_dicews_sim(sim_rctea, test_pair, reference_pairs, n, m, groups_test)
    del rctea_vecs, sim_rctea
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. LightEA & LightTEA on DICEWS-1K
    # ---------------------------------------------------------
    print('Running LightEA & LightTEA on DICEWS-1K...', flush=True)
    ee_edges, er_edges, re_edges, et_edges, te_edges = Counter(), Counter(), Counter(), Counter(), Counter()
    entity_time_sets = [set() for _ in range(total_e)]
    for h, r, t, tau in triples1 + triples2:
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

    inc = torch.zeros(total_e, total_t, dtype=torch.float32, device='cuda')
    for eid, tset in enumerate(entity_time_sets):
        if tset:
            inc[eid, list(tset)] = 1.0
    inter = inc[:n] @ inc[n:].T
    denom = inc[:n].sum(1, keepdim=True) + inc[n:].sum(1)[None, :]
    S_time_dice = torch.where(denom > 0, 2.0 * inter / denom.clamp_min(1.0), torch.zeros_like(inter))

    sim_lightea = H_rel[:n] @ H_rel[n:].T
    results['LightEA'] = evaluate_dicews_sim(sim_lightea, test_pair, reference_pairs, n, m, groups_test)

    H_ltea = F.normalize(0.7 * H_rel + 0.3 * H_time, dim=-1)
    sim_lighttea = 0.7 * (H_ltea[:n] @ H_ltea[n:].T) + 0.3 * S_time_dice
    results['LightTEA'] = evaluate_dicews_sim(sim_lighttea, test_pair, reference_pairs, n, m, groups_test)
    del A_EE, A_ER, A_RE, A_ET, A_TE, H_rel, H_time, H_ltea, sim_lightea, sim_lighttea
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 3. STEA & HTEA on DICEWS-1K
    # ---------------------------------------------------------
    print('Running STEA & HTEA on DICEWS-1K...', flush=True)
    import model_o
    import model_r

    t_int_1 = [(h, r, t, ti[tau], ti[tau]) for h, r, t, tau in triples1]
    t_int_2 = [(h, r, t, ti[tau], ti[tau]) for h, r, t, tau in triples2]
    all_tr_int = t_int_1 + t_int_2
    time1_set = {ts for _, _, _, ts, _ in t_int_1}
    time2_set = {ts for _, _, _, ts, _ in t_int_2}
    tem_valid_set = time1_set & time2_set
    all_time_int = sorted({(ts, te) for _, _, _, ts, te in all_tr_int} | {(0, 0)})
    time_int_dict_i = {pair: i for i, pair in enumerate(all_time_int)}
    zero_index = time_int_dict_i[(0, 0)]

    rec_tem_count = defaultdict(Counter)
    rec_tem_int_count = defaultdict(Counter)
    for h, r, t, ts, te in all_tr_int:
        for e in (h, t):
            rec_tem_count[e][ts] += 2
            rec_tem_int_count[e][(ts, te)] += 1

    ent_size = total_e + 1
    rel_size = total_r + 1
    time_size = total_t + 2
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
        for j, cnt in rec_tem_count[i].items():
            if j in tem_valid_set:
                time_link[i, j] = np.log1p(cnt)
        for j, cnt in rec_tem_int_count[i].items():
            if j[0] in tem_valid_set:
                time_int_link[i, time_int_dict_i[j]] = np.log1p(cnt)

    for h, r, t, ts, te in all_tr_int:
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
    adj_log = sp.coo_matrix((np.log1p(adj_coo.data), (adj_coo.row, adj_coo.col)), shape=adj_coo.shape).toarray()
    adj_features = sp.coo_matrix(adj_log / (np.sum(adj_log, axis=-1, keepdims=True) + 0.01))

    adj_idx = np.stack(adj_matrix.nonzero(), axis=1)
    rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data
    ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
    time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data
    time_int_matrix, time_int_val = np.stack(time_int_features.nonzero(), axis=1), time_int_features.data

    class DummyCfg:
        tau = 0.05
        ab_weight = 0.5
        replay = False
        neg_cross_kg = False

    device = 'cuda:0'
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    # STEA relational GNN + Dice Time Matching
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
    for _ in range(25):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * 512:(i + 1) * 512] for i in range(len(train_shuf) // 512 + 1)]:
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
        sim_stea = 0.7 * (z_stea[:n] @ z_stea[n:].T) + 0.3 * S_time_dice
        results['STEA'] = evaluate_dicews_sim(sim_stea, test_pair, reference_pairs, n, m, groups_test)
    del m_rel, opt_r, z_stea, sim_stea
    torch.cuda.empty_cache()

    # HTEA
    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    m_htea = model_o.OverAll(
        node_size=ent_size, node_hidden=100, time_hidden=50,
        rel_size=rel_features.shape[1], rel_hidden=100,
        time_size=time_size, time_int_size=time_int_size,
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
        r_neg_dis = torch.sum(torch.square(r_emb), dim=1, keepdim=True) + torch.sum(torch.square(embedding), dim=1, keepdim=True).T - 2.0 * (r_emb @ embedding.T)
        l_neg_dis = torch.sum(torch.square(l_emb), dim=1, keepdim=True) + torch.sum(torch.square(embedding), dim=1, keepdim=True).T - 2.0 * (l_emb @ embedding.T)
        mask = 1.0 - F.one_hot(left, num_classes=ent_size).float() - F.one_hot(right, num_classes=ent_size).float()
        l_loss = (pos_dis - l_neg_dis + 1.0) * mask
        r_loss = (pos_dis - r_neg_dis + 1.0) * mask
        with torch.no_grad():
            r_loss.data = (r_loss.data - r_loss.mean(-1, keepdim=True)) / r_loss.std(-1, keepdim=True).clamp_min(1e-6)
            l_loss.data = (l_loss.data - l_loss.mean(-1, keepdim=True)) / l_loss.std(-1, keepdim=True).clamp_min(1e-6)
        return torch.mean(torch.logsumexp(30.0 * l_loss + 10.0, dim=-1) + torch.logsumexp(30.0 * r_loss + 10.0, dim=-1))

    for _ in range(20):
        m_htea.train()
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * 512:(i + 1) * 512] for i in range(len(train_shuf) // 512 + 1)]:
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
        sim_htea = feat_h[:n] @ feat_h[n:].T
        results['HTEA'] = evaluate_dicews_sim(sim_htea, test_pair, reference_pairs, n, m, groups_test)
    del m_htea, opt_h, feat_h, sim_htea
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 4. MGTEA on DICEWS-1K
    # ---------------------------------------------------------
    print('Running MGTEA on DICEWS-1K...', flush=True)
    from MGTEA_models import OverAll as MGTEAOverAll
    coords = np.load(cache / 'time_coordinates.npy')
    # Coarse year bucket (every 365 days) and fine day token
    ent_mat = sp.lil_matrix((total_e, total_e))
    for i in range(total_e):
        ent_mat[i, i] = 1.0
    rel_in_m = np.zeros((total_e, total_r + 2), dtype=np.float32)
    rel_out_m = np.zeros((total_e, total_r + 2), dtype=np.float32)
    num_years = int(coords.max() // 365) + 3
    time_link_y = sp.lil_matrix((total_e, num_years))
    time_link_d = sp.lil_matrix((total_e, total_t + 2))
    radj_m = []
    for h, r, t, tau in triples1 + triples2:
        d_id = ti[tau]
        y_id = int(coords[d_id] // 365) + 1
        ent_mat[h, t] = 1
        ent_mat[t, h] = 1
        radj_m.append((h, t, r + 1, y_id, d_id))
        radj_m.append((t, h, r + 1 + total_r + 2, y_id, d_id))
        rel_out_m[h, r + 1] += 1
        rel_in_m[t, r + 1] += 1
        time_link_y[h, y_id] += 1
        time_link_y[t, y_id] += 1
        time_link_d[h, d_id] += 1
        time_link_d[t, d_id] += 1

    count = -1
    seen = set()
    deg_pair = {}
    r_index, t_index, t_indexT, r_val, adj_pairs = [], [], [], [], []
    for h, t, r_m, y_id, d_id in sorted(radj_m, key=lambda x: (x[0], x[1])):
        key = (h, t)
        if key not in seen:
            count += 1
            seen.add(key)
            deg_pair[count] = 0
            adj_pairs.append([h, t])
        deg_pair[count] += 1
        r_index.append([count, r_m])
        t_index.append([count, y_id])
        t_indexT.append([count, d_id])
        r_val.append(1.0)
    for i in range(len(r_val)):
        r_val[i] /= deg_pair[r_index[i][0]]

    adj_index = np.array(adj_pairs, dtype=np.int64)
    r_index = np.array(r_index, dtype=np.int64)
    t_index = np.array(t_index, dtype=np.int64)
    t_indexT = np.array(t_indexT, dtype=np.int64)
    r_val = np.array(r_val, dtype=np.float32)

    ent_matrix_m = np.stack(ent_mat.tocoo().nonzero(), axis=1)
    rel_matrix_m = np.stack(sp.coo_matrix(np.concatenate([rel_in_m, rel_out_m], axis=1)).nonzero(), axis=1)
    time_matrix_y = np.stack(time_link_y.tocoo().nonzero(), axis=1)
    time_matrix_d = np.stack(time_link_d.tocoo().nonzero(), axis=1)

    random.seed(2026)
    np.random.seed(2026)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    m_mgtea = MGTEAOverAll(
        node_size=total_e, node_hidden=100,
        rel_size=2 * (total_r + 2), rel_hidden=100,
        time_size=num_years, time_sizeT=total_t + 2,
        triple_size=len(adj_index),
        rel_matrix=rel_matrix_m, ent_matrix=ent_matrix_m,
        time_matrix=time_matrix_y, time_matrixT=time_matrix_d,
        dropout_rate=0.3, depth=2, device=device,
    ).to(device)
    opt_m = torch.optim.RMSprop(m_mgtea.parameters(), lr=0.005, weight_decay=0)
    inputs_m = [adj_index, r_index, r_val, t_index, rel_matrix_m, ent_matrix_m, t_indexT]

    def mgtea_loss(align_input, embedding):
        left = torch.as_tensor(align_input[:, 0], dtype=torch.long, device=device)
        right = torch.as_tensor(align_input[:, 1], dtype=torch.long, device=device)
        l_emb, r_emb = embedding[left], embedding[right]
        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        r_neg_dis = torch.sum(torch.square(r_emb), dim=1, keepdim=True) + torch.sum(torch.square(embedding), dim=1, keepdim=True).T - 2.0 * (r_emb @ embedding.T)
        l_neg_dis = torch.sum(torch.square(l_emb), dim=1, keepdim=True) + torch.sum(torch.square(embedding), dim=1, keepdim=True).T - 2.0 * (l_emb @ embedding.T)
        mask = 1.0 - F.one_hot(left, num_classes=total_e).float() - F.one_hot(right, num_classes=total_e).float()
        l_loss = (pos_dis - l_neg_dis + 1.0) * mask
        r_loss = (pos_dis - r_neg_dis + 1.0) * mask
        return torch.mean(torch.logsumexp(30.0 * l_loss + 10.0, dim=-1) + torch.logsumexp(30.0 * r_loss + 10.0, dim=-1))

    for _ in range(20):
        m_mgtea.train()
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * 512:(i + 1) * 512] for i in range(len(train_shuf) // 512 + 1)]:
            if not len(pairs):
                continue
            out = m_mgtea(inputs_m)
            loss = mgtea_loss(pairs, out)
            opt_m.zero_grad()
            loss.backward()
            opt_m.step()

    m_mgtea.eval()
    with torch.no_grad():
        emb_m = m_mgtea(inputs_m)
        emb_m = emb_m / (emb_m.norm(dim=-1, keepdim=True) + 1e-5)
        sim1 = emb_m[:n] @ emb_m[n:].T
        sim_mgtea = (sim1 + 0.5 * S_time_dice) / 1.5
        results['MGTEA'] = evaluate_dicews_sim(sim_mgtea, test_pair, reference_pairs, n, m, groups_test)

    save_json(out_dir / 'dicews_all_baselines.json', results)
    for k, v in results.items():
        f_ov = v['csls5_sinkhorn200']['full_9517x9537_test6809']['overall']
        c_ov = v['csls5_sinkhorn200']['closed_7566x7566_test6809']['overall']
        print(f"DICEWS {k:10s} | full MRR={f_ov['mrr']*100:.2f}% H@1={f_ov['hits1']*100:.2f}% H@10={f_ov['hits10']*100:.2f}% | closed MRR={c_ov['mrr']*100:.2f}%")


if __name__ == '__main__':
    main()
