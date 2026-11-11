"""Reproduce the DICEWS-1K CoTEA experiment and relation-namespace audit."""

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from reproduction.data import load_pairs, prepare, read_vocab, save_json, sha256
from reproduction.priority_hybrid_data import prepare as prepare_priority
from reproduction.priority_hybrid_data import select


INPUT_SHA256 = {
    "ent_ids_1": "4dc1158a766d8ed2026d17a48549e621bdb3d0255841d946576aed52161faaf7",
    "ent_ids_2": "dbd0076c7be918b0782d2d817b416477a89f1776692579af8bd2f0237f658d77",
    "ref_pairs": "5579ba155a0a5d21da4b78a28b6f889e3855e6dfe34769c1ec88849853571f83",
    "rel_ids_1": "f260d8a8ed0b55c79bf87e17e744026c0384ff6640493e90477735c9dab4ada4",
    "rel_ids_2": "924ad2995472c7ea9d9097ed37655e04d6f068ff9849ff608b0ce0da0aaa6f30",
    "sup_pairs": "a884456f25a5b8c446324312410c60e8cf9a124dcefb4f8e4eea897f8fdc2066",
    "triples_1": "218dac639a5109f6d2d9da3aafd8777d38954d77c1b0c051c2a15b0c6f3a0529",
    "triples_2": "e765303b2aeab8b9fd7fcb9345c8d32388e19fa84ab9d8e88a626ac4db2c049b",
}
SEED = 2026
CHUNK = 128


def prepare_inputs(data, work_dir):
    from reproduction.seed_relation_context import prepare_profiles

    for name, digest in INPUT_SHA256.items():
        if sha256(data / name) != digest:
            raise ValueError(f"DICEWS input SHA-256 did not match: {data / name}")
    for name in ("cache", "profiles"):
        if (work_dir / name).exists():
            raise FileExistsError(work_dir / name)

    work_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=work_dir) as scratch:
        scratch = Path(scratch)
        # Restrict the reference cache to the exact eight input files.
        raw = scratch / "raw" / data.name
        raw.mkdir(parents=True)
        for name in INPUT_SHA256:
            (raw / name).write_bytes((data / name).read_bytes())
        reference = scratch / "reference"
        protocol = prepare(raw, reference, length=64, seeds=1000, split_seed=SEED)
        if protocol["split"] != {"train": 1000, "dev": 757, "test": 6809}:
            raise ValueError("Unexpected DICEWS split")

        # This benchmark's verified IDs are consecutive day offsets from 2005-01-01.
        offsets = [int(value) for value in protocol["time_labels"]]
        if offsets != list(range(4017)):
            raise ValueError("Unexpected DICEWS calendar offsets")
        calendar = scratch / "calendar"
        calendar.mkdir()
        mapping = calendar / "calendar_mapping.csv"
        with mapping.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("raw_time_id", "days_since_2005_01_01"))
            writer.writerows((offset, offset) for offset in offsets)
        save_json(
            calendar / "calendar_audit.json",
            {
                "verified": True,
                "mapping_sha256": sha256(mapping),
                "origin": "2005-01-01",
                "dicews_input_sha256": INPUT_SHA256,
            },
        )

        cache = scratch / "cache"
        profiles = scratch / "profiles"
        prepare_priority(raw, reference, cache, calendar)
        prepare_profiles(raw, cache / "priority", profiles)
        cache.replace(work_dir / "cache")
        profiles.replace(work_dir / "profiles")


def train_shared(work_dir, gpu, memory_fraction, resume):
    command = [
        sys.executable, "-m", "reproduction.priority_hybrid_train",
        "--cache", str(work_dir / "cache" / "priority"),
        "--output", str(work_dir / "shared"),
        "--relation-profiles", str(work_dir / "profiles" / "seed_relation_counts.npy"),
        "--hybrid", "--relation-arm", "induced", "--seed", str(SEED),
        "--gpu", gpu, "--memory-fraction", str(memory_fraction), "--chunk", str(CHUNK),
        "--time-mask-fraction", "0.2", "--base-steps", "200", "--context-steps", "200",
        "--eval-every", "5", "--sinkhorn-every", "25",
    ]
    if resume:
        command.append("--resume")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    subprocess.run(command, env=env, check=True)


def score_pool(similarity, pairs, rows, cols):
    """Apply the paper's row-wise cosine, CSLS-5 and balanced Sinkhorn scores."""
    import torch

    from reproduction.benchmark_retrieval import balance_inplace
    from reproduction.train import average_ranks, metrics

    n, m = similarity.shape
    row_index = np.full(n, -1, dtype=np.int64)
    col_index = np.full(m, -1, dtype=np.int64)
    row_index[rows] = np.arange(len(rows))
    col_index[cols] = np.arange(len(cols))
    queries = row_index[pairs[:, 0]]
    targets = col_index[pairs[:, 1] - n]
    if (queries < 0).any() or (targets < 0).any():
        raise ValueError("The evaluation pool excludes a held-out gold pair")
    query = torch.as_tensor(queries, device=similarity.device)
    target = torch.as_tensor(targets, device=similarity.device)

    def summarize(matrix):
        blocks = [
            average_ranks(matrix[q], t).cpu().numpy()
            for q, t in zip(query.split(64), target.split(64))
        ]
        return metrics(np.concatenate(blocks))

    with torch.inference_mode():
        scores = similarity[rows][:, cols].double()
        cosine = summarize(scores)
        rx = scores.topk(5, dim=1).values.mean(1, keepdim=True)
        ry = scores.topk(5, dim=0).values.mean(0, keepdim=True)
        scores.mul_(2.0).sub_(rx).sub_(ry)
        csls = summarize(scores)
        u, v = balance_inplace(scores, temperature=0.05, iterations=200, block=256)
        scores.add_(u[:, None]).add_(v[None, :])
        return {
            "rowwise_raw_or_cosine": cosine,
            "rowwise_csls5_nosinkhorn": csls,
            "csls5_sinkhorn200": summarize(scores),
        }


def evaluate(data, work_dir, arm, device):
    import torch

    cache = work_dir / "cache" / "priority"
    with np.load(cache / "splits.npz") as splits:
        pairs = splits["test"]
        source_ids, target_ids = splits["source_raw_ids"], splits["target_raw_ids"]
    n, m = len(source_ids), len(target_ids)
    source_index = {int(value): i for i, value in enumerate(source_ids)}
    target_index = {int(value): n + i for i, value in enumerate(target_ids)}
    ref = load_pairs(data / "ref_pairs", source_index, target_index)
    arm_dir = work_dir / arm
    vectors = torch.from_numpy(np.load(arm_dir / "best_vectors.npy")).to(device)
    with torch.inference_mode():
        sim = vectors[:n] @ vectors[n:].T
        result = {
            "open": score_pool(sim, pairs, np.arange(n), np.arange(m)),
            "closed": score_pool(sim, pairs, ref[:, 0], ref[:, 1] - n),
        }
    save_json(arm_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


def disjoint_inputs(data, source_ids, target_ids, train, coordinates):
    entities = [source_ids, target_ids]
    n, m = map(len, entities)
    entity_index = [
        {int(value): offset + i for i, value in enumerate(ids)}
        for ids, offset in zip(entities, (0, n))
    ]
    relations = [sorted(read_vocab(data / f"rel_ids_{side}")) for side in (1, 2)]
    relation_index = [
        {value: offset + i for i, value in enumerate(ids)}
        for ids, offset in zip(relations, (0, len(relations[0])))
    ]
    chains = [Counter() for _ in range(n + m)]
    seed_index = {int(entity): i for i, pair in enumerate(train) for entity in pair}
    counts = np.zeros((sum(map(len, relations)), 2, len(train)), dtype=np.float64)
    for graph in (0, 1):
        for line in (data / f"triples_{graph + 1}").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, r, t, timestamp = map(int, line.split())
            h, t = entity_index[graph][h], entity_index[graph][t]
            relation = relation_index[graph][r]
            time_token = timestamp + 1
            chains[h][(t + 1, 2 * relation + 1, time_token)] += 1
            chains[t][(h + 1, 2 * relation + 2, time_token)] += 1
            if h in seed_index:
                counts[relation, 0, seed_index[h]] += 1
            if t in seed_index:
                counts[relation, 1, seed_index[t]] += 1

    events = np.zeros((n + m, 64, 4), dtype=np.int32)
    seed_objects = set((train.ravel() + 1).tolist())
    for entity, chain in enumerate(chains):
        rows = [(obj, rel, time, count) for (obj, rel, time), count in chain.items()]
        rows.sort(key=lambda row: (row[2], row[1], row[0]))
        if rows:
            chosen = select(rows, [coordinates[row[2]] for row in rows], seed_objects, set())
            events[entity, :len(chosen)] = np.asarray(rows)[chosen]
    return events, counts


def train_disjoint(data, work_dir):
    import torch

    from reproduction.chain_only import configure_attention
    from reproduction.full_context_training import cached_complete_backward
    from reproduction.priority_hybrid_train import mask_one_side_time
    from reproduction.root_mask_training import encode_root_mask
    from reproduction.seed_relation_context import SeedRelationModel

    output = work_dir / "disjoint"
    output.mkdir(parents=True)
    cache = work_dir / "cache" / "priority"
    protocol = json.loads((cache / "protocol.json").read_text(encoding="utf-8"))
    n, m = protocol["entities"]
    coordinates = np.load(cache / "time_coordinates.npy")
    with np.load(cache / "splits.npz") as splits:
        train, dev = splits["train"], splits["dev"]
        source_ids, target_ids = splits["source_raw_ids"], splits["target_raw_ids"]
    events, counts = disjoint_inputs(data, source_ids, target_ids, train, coordinates)
    protocol["shared_relations"] = len(counts)

    configure_attention()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = SeedRelationModel(protocol, events, train, counts, coordinates, "induced").cuda()
    extra = model.encoder.extra_parameters()
    extra_ids = {id(param) for param in extra}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in extra_ids]
    model.encoder.use_neighbor = False
    for parameter in extra:
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(backbone, lr=1e-4, weight_decay=1e-4)
    best_mrr = -1.0
    for step in range(1, 351):
        if step == 201:
            model.encoder.use_neighbor = True
            for parameter in extra:
                parameter.requires_grad_(True)
            optimizer.add_param_group({"params": extra, "lr": 5e-4, "weight_decay": 1e-4})
        model.train()
        optimizer.zero_grad(set_to_none=True)
        view, _ = mask_one_side_time(events, train, 0.2, SEED, step)
        cached_complete_backward(model, view, train, chunk=CHUNK)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()

        if step in (250, 300, 325, 350):
            vectors = encode_root_mask(model, events, CHUNK).detach().cpu()
            left, right = vectors[:n].cuda().float(), vectors[n:].cuda().float()
            sim = left @ right.T
            result = score_pool(sim, dev, np.arange(n), np.arange(m))
            mrr = result["csls5_sinkhorn200"]["mrr"]
            print(f"DICEWS disjoint, step {step}: validation Sinkhorn MRR {mrr:.4%}", flush=True)
            if mrr > best_mrr:
                best_mrr = mrr
                np.save(output / "best_vectors.npy", vectors.numpy())
                save_json(output / "validation.json", {"step": step, "metrics": result})
            del sim, left, right, vectors
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "disjoint"))
    parser.add_argument("--data", type=Path, default=Path("datasets/ICEWS05-15"))
    parser.add_argument("--work-dir", type=Path, default=Path("runs/dicews"))
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--memory-fraction", type=float, default=0.12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--arm", choices=("shared", "disjoint"), default="shared")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare_inputs(args.data, args.work_dir)
    elif args.command == "train":
        train_shared(args.work_dir, args.gpu, args.memory_fraction, args.resume)
    elif args.command == "evaluate":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        evaluate(args.data, args.work_dir, args.arm, args.device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        train_disjoint(args.data, args.work_dir)
        evaluate(args.data, args.work_dir, "disjoint", "cuda")


if __name__ == "__main__":
    main()
