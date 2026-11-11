"""Run STEA (2022), MGTEA (2024), HTEA (2025), and RCTEA (2026) on WUXI-TEMP."""
from collections import Counter, defaultdict
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
sys.path.insert(0, str(RCTEA_ROOT))

from reproduction.benchmark_retrieval import balance_inplace
from reproduction.data import read_vocab, save_json
from reproduction.train import average_ranks


def prepare_contiguous_wuxi(raw_dir, suite_dir):
    out_data = suite_dir / 'wuxi_int_dataset'
    if (out_data / 'READY.json').exists():
        return out_data
    out_data.mkdir(parents=True, exist_ok=True)
    cache = suite_dir / 'cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']

    ev = [read_vocab(raw_dir / f'ent_ids_{i}') for i in (1, 2)]
    raw_ids = [sorted(v) for v in ev]
    indices = [{v: j + off for j, v in enumerate(ids)} for ids, off in zip(raw_ids, (0, n))]
    rv = [read_vocab(raw_dir / f'rel_ids_{i}') for i in (1, 2)]
    ri = {label: i for i, label in enumerate(protocol['relation_names'])}
    ti = {label: int(k) for k, label in protocol['time_labels'].items()}
    ti['~'] = 0

    with (out_data / 'ent_ids_1').open('w', encoding='utf-8') as f:
        for i in range(n):
            f.write(f'{i}\te1_{i}\n')
    with (out_data / 'ent_ids_2').open('w', encoding='utf-8') as f:
        for j in range(m):
            f.write(f'{n + j}\te2_{j}\n')

    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train = splits['train']
        reference = splits['reference']
    np.savetxt(out_data / 'sup_pairs', train, fmt='%d', delimiter='\t')
    np.savetxt(out_data / 'ref_pairs', reference, fmt='%d', delimiter='\t')

    for g in range(2):
        with (out_data / f'triples_{g + 1}').open('w', encoding='utf-8') as f_int, \
             (out_data / f'triples_str_{g + 1}').open('w', encoding='utf-8') as f_str:
            for line in (raw_dir / f'triples_{g + 1}').read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                h_raw, r_raw, t_raw, start, end = line.split('\t')
                h = indices[g][int(h_raw)]
                t = indices[g][int(t_raw)]
                r = ri[rv[g][int(r_raw)]]
                s_id = ti[start]
                e_id = ti[end]
                f_int.write(f'{h}\t{r}\t{t}\t{s_id}\t{e_id}\n')
                f_str.write(f'{h}\t{r}\t{t}\t{start}\t{end}\n')

    save_json(out_data / 'READY.json', {'entities': [n, m], 'train': len(train), 'ref': len(reference)})
    return out_data


@torch.inference_mode()
def evaluate_sim_matrix(sim_n_by_m, reference, candidates_eval, candidates_all, n, total_e, groups):
    """Evaluate an (n, m) similarity matrix under both native Sinkhorn10 and unified CSLS5+Sinkhorn200."""
    def eval_pool(pool_rows, pool_cols, mode='csls5_sinkhorn200'):
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
        q = torch.as_tensor(rowmap[reference[:, 0]], device='cuda')
        g = torch.as_tensor(colmap[reference[:, 1]], device='cuda')
        r = average_ranks(sub[q], g).cpu().numpy()

        def m_fn(arr):
            return dict(
                mrr=float((1.0 / arr).mean()),
                hits1=float((arr <= 1.0).mean()),
                hits10=float((arr <= 10.0).mean()),
                queries=int(len(arr)),
            )
        return dict(
            overall=m_fn(r),
            groups={k: m_fn(r[mask]) for k, mask in groups.items()},
        )

    out = {}
    for mode in ('native_sinkhorn10', 'csls5_sinkhorn200'):
        out[mode] = dict(
            eval_5483=eval_pool(reference[:, 0], candidates_eval, mode),
            closed_3872=eval_pool(reference[:, 0], reference[:, 1], mode),
            full_6483=eval_pool(np.arange(n), candidates_all, mode),
        )
    return out


def build_rctea_htea_matrices(int_data_dir, total_e, total_r, total_t, valid_int_override=None):
    """Build exact feature & adjacency matrices as in RCTEA utils_n.py / HTEA utils.py."""
    triples1, triples2 = [], []
    time1, time2 = set(), set()
    time_int1, time_int2 = set(), set()

    for g, t_list, t_set, ti_set in [(1, triples1, time1, time_int1), (2, triples2, time2, time_int2)]:
        for line in (int_data_dir / f'triples_{g}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h, r, t, ts, te = map(int, line.split('\t'))
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
    tem_int_valid_set = valid_int_override if valid_int_override is not None else (time_int1 & time_int2)
    all_time_int = sorted(time_int1 | time_int2)
    if (0, 0) not in all_time_int:
        all_time_int.insert(0, (0, 0))
    time_int_dict_i = {ti: i for i, ti in enumerate(all_time_int)}
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
    time_size = total_t + 1
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

    radj = []
    for h, r, t, ts, te in all_triples:
        t_idx = time_int_dict_i[(ts, te)]
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] += 1
        adj_features[t, h] += 1
        adj_features[h, -1] += 1
        adj_features[t, -1] += 1
        radj.append([h, t, r, t_idx])
        radj.append([t, h, r + rel_size, t_idx])
        rel_in[h, r] += 1
        rel_out[t, r] += 1
        rel_in[h, total_r] += 1
        rel_out[t, total_r] += 1

    radj.append([total_e, total_e, 0, 0])
    adj_matrix[total_e, total_e] = 1

    time_features = sp.coo_matrix(time_link / (np.sum(time_link, axis=-1, keepdims=True) + 0.01))
    time_int_features = sp.coo_matrix(time_int_link / (np.sum(time_int_link, axis=-1, keepdims=True) + 0.01))

    rel_concat = np.log1p(np.concatenate([rel_in, rel_out], axis=1))
    rel_features = sp.coo_matrix(rel_concat / (np.sum(rel_concat, axis=-1, keepdims=True) + 0.01))

    adj_coo = adj_features.tocoo()
    adj_log = sp.coo_matrix((np.log1p(adj_coo.data), (adj_coo.row, adj_coo.col)), shape=adj_coo.shape).toarray()
    adj_features = sp.coo_matrix(adj_log / (np.sum(adj_log, axis=-1, keepdims=True) + 0.01))

    return (
        adj_matrix, adj_features, rel_features, time_features, time_int_features,
        radj, time_int_dict_i, triples1, triples2,
    )


def run_rctea_and_stea(int_data_dir, suite_dir, seed=2026):
    """Run RCTEA (4 stages) and STEA (Relational GNN + Dice Time Matching) on WUXI-TEMP."""
    import model_o
    import model_r
    import model_t
    from models import RelCNN
    from dgl_dual import DGMC

    cache = suite_dir / 'cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m
    total_r = protocol['shared_relations']
    total_t = protocol['time_tokens']

    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train_pair = splits['train'].copy()
        dev_pair = splits['reference'].copy()
        candidates_eval = splits['candidates_eval']
        candidates_all = splits['candidates_all']
    groups = dict(np.load(suite_dir / 'wuxi_groups.npz'))

    (
        adj_matrix, adj_features, rel_features, time_features, time_int_features,
        radj, time_int_dict_i, triples1, triples2,
    ) = build_rctea_htea_matrices(int_data_dir, total_e, total_r, total_t)

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

    class DummyCfg:
        tau = 0.05
        ab_weight = 0.5
        replay = False
        neg_cross_kg = False
        random_seed = seed

    cfgs = DummyCfg()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cuda:0'
    node_hidden = 50
    rel_hidden = 50
    time_hidden = 50
    batch_size = 512
    dropout_rate = 0.3
    lr = 0.005
    depth = 2

    # Stage 1: Relational GNN (model_rel, 25 epochs)
    t0 = time.perf_counter()
    model_rel = model_r.OverAll(
        node_size=node_size, node_hidden=node_hidden,
        rel_size=rel_size, rel_hidden=rel_hidden,
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val, args=cfgs,
        triple_size=triple_size, dropout_rate=dropout_rate,
        depth=depth, device=device,
    ).to(device)
    opt_r = torch.optim.RMSprop(model_rel.parameters(), lr=lr, weight_decay=0)
    train_shuf = train_pair.copy()
    for _ in range(25):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            _, loss_ent = model_rel([adj_idx, pairs])
            loss_ent.backward(retain_graph=True)
            opt_r.step()
            opt_r.zero_grad()

    # Evaluate STEA right here: Relational GNN embeddings + Dice Time Similarity
    model_rel.eval()
    model_rel.return_loss = False
    with torch.no_grad():
        z_rel = model_rel([adj_idx, train_pair])[:total_e]
        z_rel = F.normalize(z_rel, dim=-1)
        sim_rel = z_rel[:n] @ z_rel[n:].T

        # Build Dice time similarity matrix (omitting missing time 0)
        inc = torch.zeros(total_e, total_t + 1, dtype=torch.float32, device=device)
        for h, r, t, ts, te in triples1 + triples2:
            if ts > 0:
                inc[h, ts] = 1.0
                inc[t, ts] = 1.0
        inc_l, inc_r = inc[:n, 1:], inc[n:, 1:]
        inter = inc_l @ inc_r.T
        denom = inc_l.sum(1, keepdim=True) + inc_r.sum(1)[None, :]
        sim_time_dice = torch.where(denom > 0, 2.0 * inter / denom.clamp_min(1.0), torch.zeros_like(inter))
        sim_stea = 0.7 * sim_rel + 0.3 * sim_time_dice
        stea_res = evaluate_sim_matrix(sim_stea, dev_pair, candidates_eval, candidates_all, n, total_e, groups)
        stea_res['elapsed_seconds'] = time.perf_counter() - t0

    # Stage 2: Temporal GNN (model_time, 60 epochs)
    model_time = model_t.OverAll(
        node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
        rel_size=rel_size, rel_hidden=rel_hidden,
        time_size=time_size, time_int_size=time_int_size,
        time_matrix=time_matrix, time_val=time_val,
        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
        triple_size=triple_size, dropout_rate=dropout_rate,
        depth=depth, device=device,
    ).to(device)
    model_time.zero_interval = time_int_dict_i[(0, 0)]

    opt_t = torch.optim.RMSprop(model_time.parameters(), lr=lr, weight_decay=0)
    for _ in range(60):
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            _, loss_time = model_time([adj_idx, pairs])
            loss_time.backward(retain_graph=True)
            opt_t.step()
            opt_t.zero_grad()

    del opt_r, opt_t
    torch.cuda.empty_cache()

    # Stage 3: DGMC Neighborhood Consensus (90 epochs)
    all_pair = np.vstack((train_pair, dev_pair))
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
    dgmc = DGMC(model_time, model_rel, psi_3, num_steps=0, k=15).to(device)
    opt_dgmc = torch.optim.Adam(dgmc.parameters(), lr=0.001)
    N_s = len(all_pair)
    inputs_full = [adj_idx, train_pair]

    for ep in range(1, 91):
        if ep == 31:
            dgmc.num_steps = 5
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

    # Stage 4: Joint Model (model_over, 5 epochs)
    model_over = model_o.OverAll(
        node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
        rel_size=rel_size, rel_hidden=rel_hidden,
        time_size=time_size, time_int_size=time_int_size,
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val,
        time_matrix=time_matrix, time_val=time_val, out_dim=4,
        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=cfgs,
        triple_size=triple_size, dropout_rate=dropout_rate,
        depth=depth, device=device,
    ).to(device)
    model_over.zero_interval = time_int_dict_i[(0, 0)]

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

    opt_o = torch.optim.RMSprop(model_over.parameters(), lr=lr, weight_decay=0)
    for _ in range(5):
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
    with torch.no_grad():
        feat_rctea = model_over([adj_idx, train_pair])[:total_e]
        feat_rctea = feat_rctea / (feat_rctea.norm(dim=-1, keepdim=True) + 1e-5)
        sim_rctea = feat_rctea[:n] @ feat_rctea[n:].T
        rctea_res = evaluate_sim_matrix(sim_rctea, dev_pair, candidates_eval, candidates_all, n, total_e, groups)
        rctea_res['elapsed_seconds'] = time.perf_counter() - t0
        np.save(suite_dir / 'rctea_wuxi_embeddings.npy', feat_rctea.cpu().numpy())

    return stea_res, rctea_res


def run_htea(int_data_dir, suite_dir, seed=2026):
    """Run HTEA (WSDM 2025: standardized LogSumExp margin loss + heterogeneity interval refinement)."""
    import model_o

    cache = suite_dir / 'cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m
    total_r = protocol['shared_relations']
    total_t = protocol['time_tokens']

    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train_pair = splits['train'].copy()
        dev_pair = splits['reference'].copy()
        candidates_eval = splits['candidates_eval']
        candidates_all = splits['candidates_all']
    groups = dict(np.load(suite_dir / 'wuxi_groups.npz'))

    (
        adj_matrix, adj_features, rel_features, time_features, time_int_features,
        radj, time_int_dict_i, triples1, triples2,
    ) = build_rctea_htea_matrices(int_data_dir, total_e, total_r, total_t)

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

    class DummyCfg:
        tau = 0.05
        ab_weight = 0.5
        replay = False
        neg_cross_kg = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = 'cuda:0'
    node_hidden = 100
    rel_hidden = 100
    time_hidden = 50
    batch_size = 512
    dropout_rate = 0.3
    lr = 0.005
    gamma = 1.0
    depth = 2

    def htea_align_loss(align_input, embedding):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1, keepdim=True)
            row_norms_B = torch.sum(torch.square(B), dim=1, keepdim=True).T
            return row_norms_A + row_norms_B - 2.0 * (A @ B.T)

        left = torch.as_tensor(align_input[:, 0], dtype=torch.long, device=device)
        right = torch.as_tensor(align_input[:, 1], dtype=torch.long, device=device)
        l_emb = embedding[left]
        r_emb = embedding[right]
        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        r_neg_dis = squared_dist(r_emb, embedding)
        l_neg_dis = squared_dist(l_emb, embedding)

        mask = (1.0 - F.one_hot(left, num_classes=node_size).float() - F.one_hot(right, num_classes=node_size).float())
        l_loss = (pos_dis - l_neg_dis + gamma) * mask
        r_loss = (pos_dis - r_neg_dis + gamma) * mask

        with torch.no_grad():
            r_mean = torch.mean(r_loss, dim=-1, keepdim=True)
            r_std = torch.std(r_loss, dim=-1, keepdim=True).clamp_min(1e-6)
            r_loss.data = (r_loss.data - r_mean) / r_std
            l_mean = torch.mean(l_loss, dim=-1, keepdim=True)
            l_std = torch.std(l_loss, dim=-1, keepdim=True).clamp_min(1e-6)
            l_loss.data = (l_loss.data - l_mean) / l_std

        lamb, tau = 30.0, 10.0
        return torch.mean(torch.logsumexp(lamb * l_loss + tau, dim=-1) + torch.logsumexp(lamb * r_loss + tau, dim=-1))

    t0 = time.perf_counter()
    model = model_o.OverAll(
        node_size=node_size, node_hidden=node_hidden, time_hidden=time_hidden,
        rel_size=rel_size, rel_hidden=rel_hidden,
        time_size=time_size, time_int_size=time_int_size,
        rel_matrix=rel_matrix, rel_val=rel_val,
        ent_matrix=ent_matrix, ent_val=ent_val,
        time_matrix=time_matrix, time_val=time_val, out_dim=4,
        time_int_matrix=time_int_matrix, time_int_val=time_int_val, args=DummyCfg(),
        triple_size=triple_size, dropout_rate=dropout_rate,
        depth=depth, device=device,
    ).to(device)
    model.zero_interval = time_int_dict_i[(0, 0)]
    model.return_loss = False
    opt = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0)
    train_shuf = train_pair.copy()

    # HTEA trains 20 epochs
    for _ in range(20):
        model.train()
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            out_ent = model([adj_idx, pairs])
            loss = htea_align_loss(pairs, out_ent)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        feat = model([adj_idx, train_pair])[:total_e]
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-5)
        sim_htea = feat[:n] @ feat[n:].T
        htea_res = evaluate_sim_matrix(sim_htea, dev_pair, candidates_eval, candidates_all, n, total_e, groups)
        htea_res['elapsed_seconds'] = time.perf_counter() - t0
        np.save(suite_dir / 'htea_wuxi_embeddings.npy', feat.cpu().numpy())

    return htea_res


def run_mgtea(int_data_dir, suite_dir, seed=2026):
    """Run MGTEA (CIKM 2024: Dual-Granularity Time GNN + 3-Hop Coarse/Fine Time Sim Fusion)."""
    mgtea_dir = ROOT_EXP / 'third_party/mgtea'
    sys.path.insert(0, str(mgtea_dir))
    from MGTEA_models import OverAll as MGTEAOverAll

    cache = suite_dir / 'cache/priority'
    protocol = json.loads((cache / 'protocol.json').read_text(encoding="utf-8"))
    n, m = protocol['entities']
    total_e = n + m
    total_r = protocol['shared_relations']

    with np.load(suite_dir / 'reference/splits.npz') as splits:
        train_pair = splits['train'].copy()
        dev_pair = splits['reference'].copy()
        candidates_eval = splits['candidates_eval']
        candidates_all = splits['candidates_all']
    groups = dict(np.load(suite_dir / 'wuxi_groups.npz'))

    # Parse coarse year and fine date tokens as in MGTEA_utils.py / MGTEA_utils2.py
    year_vocab = {'~': 0}
    fine_vocab = {'~': 0}
    raw_triples = []
    for g in (1, 2):
        for line in (int_data_dir / f'triples_str_{g}').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            h_s, r_s, t_s, ts, te = line.split('\t')
            h, r, t = int(h_s), int(r_s), int(t_s)
            y_id, f_id = 0, 0
            if ts != '~' and '.' in ts:
                y_part, md_part = ts.split('.')
                if y_part not in year_vocab:
                    year_vocab[y_part] = len(year_vocab)
                y_id = year_vocab[y_part]
                # MGTEA_utils2.py lines 43, 115: filters out 0101, 0100, 0000 in fine branch
                if md_part not in ('0101', '0100', '0000'):
                    if ts not in fine_vocab:
                        fine_vocab[ts] = len(fine_vocab)
                    f_id = fine_vocab[ts]
            raw_triples.append((h, r + 1, t, y_id, f_id))

    ent_size = total_e
    rel_size = 2 * (total_r + 2)
    time_size = len(year_vocab) + 1
    time_sizeT = len(fine_vocab) + 1

    adj_mat = sp.lil_matrix((ent_size, ent_size))
    ent_mat = sp.lil_matrix((ent_size, ent_size))
    for i in range(ent_size):
        ent_mat[i, i] = 1.0
    rel_in = np.zeros((ent_size, total_r + 2), dtype=np.float32)
    rel_out = np.zeros((ent_size, total_r + 2), dtype=np.float32)
    time_link = sp.lil_matrix((ent_size, time_size))
    time_linkT = sp.lil_matrix((ent_size, time_sizeT))

    radj = []
    for h, r, t, y_id, f_id in raw_triples:
        adj_mat[h, t] = 1
        adj_mat[t, h] = 1
        ent_mat[h, t] = 1
        ent_mat[t, h] = 1
        radj.append((h, t, r, y_id, f_id))
        radj.append((t, h, r + total_r + 2, y_id, f_id))
        rel_out[h, r] += 1
        rel_in[t, r] += 1
        time_link[h, y_id] += 1
        time_link[t, y_id] += 1
        time_linkT[h, f_id] += 1
        time_linkT[t, f_id] += 1

    count = -1
    seen = set()
    deg_pair = {}
    r_index, t_index, t_indexT, r_val = [], [], [], []
    adj_pairs = []
    for h, t, r, y_id, f_id in sorted(radj, key=lambda x: (x[0], x[1])):
        key = (h, t)
        if key not in seen:
            count += 1
            seen.add(key)
            deg_pair[count] = 0
            adj_pairs.append([h, t])
        deg_pair[count] += 1
        r_index.append([count, r])
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
    rel_features = sp.coo_matrix(np.concatenate([rel_in, rel_out], axis=1))
    rel_matrix = np.stack(rel_features.nonzero(), axis=1)
    time_matrix = np.stack(time_link.tocoo().nonzero(), axis=1)
    time_matrixT = np.stack(time_linkT.tocoo().nonzero(), axis=1)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = 'cuda:0'

    t0 = time.perf_counter()
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

    def mgtea_align_loss(align_input, embedding):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1, keepdim=True)
            row_norms_B = torch.sum(torch.square(B), dim=1, keepdim=True).T
            return row_norms_A + row_norms_B - 2.0 * (A @ B.T)

        left = torch.as_tensor(align_input[:, 0], dtype=torch.long, device=device)
        right = torch.as_tensor(align_input[:, 1], dtype=torch.long, device=device)
        l_emb = embedding[left]
        r_emb = embedding[right]
        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        r_neg_dis = squared_dist(r_emb, embedding)
        l_neg_dis = squared_dist(l_emb, embedding)
        mask = (1.0 - F.one_hot(left, num_classes=ent_size).float() - F.one_hot(right, num_classes=ent_size).float())
        l_loss = (pos_dis - l_neg_dis + 1.0) * mask
        r_loss = (pos_dis - r_neg_dis + 1.0) * mask
        return torch.mean(torch.logsumexp(30.0 * l_loss + 10.0, dim=-1) + torch.logsumexp(30.0 * r_loss + 10.0, dim=-1))

    train_shuf = train_pair.copy()
    batch_size = 512
    inputs = [adj_index, r_index, r_val, t_index, rel_matrix, ent_matrix, t_indexT]
    for _ in range(20):
        model.train()
        np.random.shuffle(train_shuf)
        for pairs in [train_shuf[i * batch_size:(i + 1) * batch_size] for i in range(len(train_shuf) // batch_size + 1)]:
            if not len(pairs):
                continue
            out = model(inputs)
            loss = mgtea_align_loss(pairs, out)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        emb = model(inputs)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-5)
        sim1 = emb[:n] @ emb[n:].T

        # MGTEA explicit coarse (year) and fine (day) 3-hop propagated time similarities (MGTEA_seu_tkg.py)
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

        def prop_time_sim(t_mat_lil, num_t):
            dense = torch.tensor(t_mat_lil.toarray(), dtype=torch.float32, device=device)
            # Zero out missing column 0
            dense[:, 0] = 0.0
            feat = F.normalize(dense, dim=-1)
            sim = feat[:n] @ feat[n:].T
            for _ in range(3):
                feat = F.normalize(torch.sparse.mm(A_rel, feat), dim=-1)
                sim = sim + (feat[:n] @ feat[n:].T)
            return sim / 4.0

        time_sim_year = prop_time_sim(time_link, time_size)
        time_sim_day = prop_time_sim(time_linkT, time_sizeT)
        sim4 = 0.85 * time_sim_year + 0.15 * time_sim_day
        sim_mgtea = (sim1 + 0.5 * sim4) / 1.5
        mgtea_res = evaluate_sim_matrix(sim_mgtea, dev_pair, candidates_eval, candidates_all, n, total_e, groups)
        mgtea_res['elapsed_seconds'] = time.perf_counter() - t0
        np.save(suite_dir / 'mgtea_wuxi_embeddings.npy', emb.cpu().numpy())

    return mgtea_res


def main():
    raw_dir = ROOT_EXP / 'datasets/WUXI-TEMP'
    suite_dir = ROOT_EXP / 'experiments/wuxi_suite_20261108'
    int_data_dir = prepare_contiguous_wuxi(raw_dir, suite_dir)

    print('Running STEA and RCTEA...', flush=True)
    stea_res, rctea_res = run_rctea_and_stea(int_data_dir, suite_dir)
    print('STEA done:', stea_res['csls5_sinkhorn200']['eval_5483']['overall'], flush=True)
    print('RCTEA done:', rctea_res['csls5_sinkhorn200']['eval_5483']['overall'], flush=True)

    print('Running HTEA...', flush=True)
    htea_res = run_htea(int_data_dir, suite_dir)
    print('HTEA done:', htea_res['csls5_sinkhorn200']['eval_5483']['overall'], flush=True)

    print('Running MGTEA...', flush=True)
    mgtea_res = run_mgtea(int_data_dir, suite_dir)
    print('MGTEA done:', mgtea_res['csls5_sinkhorn200']['eval_5483']['overall'], flush=True)

    payload = dict(
        state='completed_four_baselines',
        STEA=stea_res,
        MGTEA=mgtea_res,
        HTEA=htea_res,
        RCTEA=rctea_res,
    )
    save_json(suite_dir / 'four_baselines_results.json', payload)
    print('ALL FOUR BASELINES COMPLETED!', flush=True)


if __name__ == '__main__':
    main()
