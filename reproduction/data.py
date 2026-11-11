"""LightTEA data acquisition and deterministic, auditable event-chain preparation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np

UPSTREAM = "https://github.com/lcai2/LightTEA"
ARCHIVE_URL = "https://raw.githubusercontent.com/lcai2/LightTEA/main/data.zip"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(destination, archive=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive = Path(archive) if archive else destination / "lighttea_data.zip"
    if not archive.exists():
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            relative = Path(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.is_dir():
                continue
            parts = relative.parts
            if len(parts) != 3 or parts[0] != "data":
                continue
            target = destination / parts[1] / parts[2]
            if target.exists():
                if hashlib.sha256(source.read(member)).hexdigest() != sha256(target):
                    raise ValueError(f"Existing file differs from upstream: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read(member))
    manifest = {"repository": UPSTREAM, "archive_url": ARCHIVE_URL,
                "archive_sha256": sha256(archive),
                "files": {str(p.relative_to(destination)): sha256(p)
                          for p in sorted(destination.glob("*/*")) if p.is_file()}}
    save_json(destination / "source_manifest.json", manifest)
    return manifest


def read_vocab(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        left, right = line.split("\t", 1)
        key, label = (int(left), right.strip()) if left.lstrip("-").isdigit() else (int(right), left.strip())
        if key in result:
            raise ValueError(f"Duplicate ID in {path}: {key}")
        result[key] = label
    return result


def load_pairs(path, source_index, target_index):
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            left, right = map(int, line.split())
            pairs.append((source_index[left], target_index[right]))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def partition_pairs(pairs, seeds, split_seed):
    if not 0 < seeds < len(pairs) - 1:
        raise ValueError("Training seed count leaves no evaluation pairs")
    if any(len(np.unique(pairs[:, col])) != len(pairs) for col in (0, 1)):
        raise ValueError("Alignment pairs must be unique and one-to-one")
    training = pairs[:seeds]
    remaining = pairs[seeds:]
    permutation = np.random.default_rng(split_seed).permutation(len(remaining))
    n_dev = max(1, round(0.1 * len(remaining)))
    dev = remaining[permutation[:n_dev]]
    test = remaining[permutation[n_dev:]]
    assert not set(training[:, 0]) & set(dev[:, 0])
    assert not set(training[:, 0]) & set(test[:, 0])
    assert not set(dev[:, 0]) & set(test[:, 0])
    return training, dev, test


def prepare(data_dir, output, length=64, seeds=1000, split_seed=2026):
    data_dir, output = Path(data_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entity_vocab = [read_vocab(data_dir / f"ent_ids_{i}") for i in (1, 2)]
    counts = [len(v) for v in entity_vocab]
    raw_ids = [sorted(v) for v in entity_vocab]
    # Graph identity is part of the key, even if the raw ID spaces overlap.
    indices = [{value: j + offset for j, value in enumerate(raw)}
               for raw, offset in zip(raw_ids, (0, counts[0]))]
    relation_vocab = [read_vocab(data_dir / f"rel_ids_{i}") for i in (1, 2)]
    # Only exact schema labels are shared; entity labels never become features.
    relation_names = sorted(set(relation_vocab[0].values()) | set(relation_vocab[1].values()))
    relation_index = {label: i for i, label in enumerate(relation_names)}
    canonical = [{r: relation_index[label] for r, label in vocab.items()} for vocab in relation_vocab]
    fact_rows, time_ids, fact_counts = [], set(), []
    interval = False
    for graph in range(2):
        count = 0
        with (data_dir / f"triples_{graph + 1}").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = list(map(int, line.split()))
                if len(row) not in (4, 5):
                    raise ValueError(f"Unsupported fact: {row}")
                head, rel, tail = row[:3]
                interval |= len(row) == 5
                times = tuple(dict.fromkeys(row[3:]))
                fact_rows.append((indices[graph][head], canonical[graph][rel], indices[graph][tail], times))
                time_ids.update(times)
                count += 1
        fact_counts.append(count)
    time_path = data_dir / "time_id"
    time_labels = read_vocab(time_path) if time_path.exists() else None
    if interval and time_labels is None:
        raise ValueError("Interval-time data requires a time_id vocabulary")
    missing = set()
    if time_labels:
        missing = {k for k, v in time_labels.items() if v.lower() in ("-inf", "inf", "+inf", "null", "none")}
        known = sorted(time_ids - missing, key=lambda k: (float(time_labels[k]), k))
        chronology = "numeric year labels from upstream time_id; interval endpoints become point events"
    else:
        known = sorted(time_ids)
        chronology = "numeric timestamp-ID order ASSUMPTION; upstream provides no date mapping; calendar chronology unverified"
    time_index = {value: i + 1 for i, value in enumerate(known)}
    chains = [Counter() for _ in range(sum(counts))]
    for head, rel, tail, raw_times in fact_rows:
        times = [time_index[t] for t in raw_times if t not in missing] or [0]
        for timestamp in times:
            chains[head][(tail + 1, 2 * rel + 1, timestamp)] += 1
            chains[tail][(head + 1, 2 * rel + 2, timestamp)] += 1
    del fact_rows
    events = np.zeros((sum(counts), length, 4), dtype=np.int32)
    lengths = np.asarray([len(c) for c in chains], dtype=np.int32)
    for entity, chain in enumerate(chains):
        rows = sorted(((obj, rel, t, freq) for (obj, rel, t), freq in chain.items()),
                      key=lambda row: (row[2] == 0, row[2], row[1], row[0]))
        # Cover the full observed history rather than retaining only the latest window.
        if len(rows) > length:
            positions = np.linspace(0, len(rows) - 1, length, dtype=np.int64)
            rows = [rows[i] for i in positions]
        if rows:
            events[entity, :len(rows)] = rows
    del chains
    # Read alignment labels only after all event features have been constructed.
    pairs = np.concatenate([load_pairs(data_dir / name, *indices) for name in ("sup_pairs", "ref_pairs")])
    train, dev, test = partition_pairs(pairs, seeds, split_seed)
    target_all = np.arange(counts[0], sum(counts), dtype=np.int64)
    nontrain_candidates = np.setdiff1d(target_all, train[:, 1])
    np.savez_compressed(output / "events.npz", events=events, lengths=lengths)
    # LightTEA-style restricted candidate list is fixed before the dev/test split.
    np.savez(output / "splits.npz", train=train, dev=dev, test=test,
             candidates_all=target_all, candidates_nontrain=nontrain_candidates,
             candidates_reference=np.sort(pairs[seeds:, 1]),
             source_raw_ids=np.asarray(raw_ids[0]), target_raw_ids=np.asarray(raw_ids[1]))
    protocol = {"implementation": "independent reconstruction from manuscript; no prior implementation used",
                "dataset": data_dir.name, "entities": counts, "facts": fact_counts,
                "relations": [len(v) for v in relation_vocab], "shared_relations": len(relation_names),
                "relation_names": relation_names, "time_tokens": len(time_index),
                "time_labels": {str(t): time_labels[t] if time_labels else str(t) for t in known},
                "chronology": chronology, "intervals": interval,
                "missing_time": "masked time slot; preserve object/relation/frequency evidence",
                "max_length": length, "truncation": "evenly spaced over ordered aggregated events",
                "same_time_order": "relation token then graph-local object ID; arbitrary deterministic tie break",
                "object_features": "graph-local learned entity-ID embeddings; no entity names or test aliases",
                "relation_features": "shared only by exact relation label",
                "training_seeds": seeds, "split_seed": split_seed,
                "split": {"train": len(train), "dev": len(dev), "test": len(test)},
                "candidates": {"all": len(target_all), "nontrain": len(nontrain_candidates), "reference": len(pairs) - seeds},
                "chain_lengths": {"median": float(np.median(lengths)), "max": int(lengths.max()),
                                  "empty": int((lengths == 0).sum()), "truncated": int((lengths > length).sum())},
                "source_sha256": {p.name: sha256(p) for p in sorted(data_dir.iterdir()) if p.is_file()}}
    save_json(output / "protocol.json", protocol)
    print(json.dumps(protocol, ensure_ascii=False), flush=True)
    return protocol


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download")
    fetch.add_argument("--destination", default="datasets/lighttea")
    fetch.add_argument("--archive")
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--length", type=int, default=64)
    prep.add_argument("--seeds", type=int, default=1000)
    prep.add_argument("--split-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "download":
        print(json.dumps(download(args.destination, args.archive), indent=2))
    else:
        if args.length < 1:
            raise ValueError("length must be positive")
        prepare(args.data, args.output, args.length, args.seeds, args.split_seed)


if __name__ == "__main__":
    main()
