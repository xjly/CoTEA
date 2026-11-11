"""Sampled-softmax training and chunked exact cosine retrieval, without test selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import save_json, sha256
from .model import EventChainAligner


def pair_scores(left, right, temperature, score_dtype="float32"):
    if score_dtype == "float32":
        with torch.autocast(left.device.type, enabled=False):
            return torch.einsum("bd,bkd->bk", left.float(), right.float()) / temperature
    return torch.einsum("bd,bkd->bk", left, right) / temperature


def sampled_loss(model, events, source, targets, device, temperature, amp, score_dtype="float32",
                 teacher=None, teacher_weight=0.0):
    requested = np.concatenate([source, targets.ravel()])
    unique, inverse = np.unique(requested, return_inverse=True)
    inputs = torch.as_tensor(events[unique], dtype=torch.long, device=device)
    inverse = torch.as_tensor(inverse, device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
        embedding = model(inputs, central_ids=torch.as_tensor(unique, device=device))
        left = embedding[inverse[:len(source)]]
        right = embedding[inverse[len(source):]].reshape(len(source), targets.shape[1], embedding.shape[-1])
        score = pair_scores(left, right, temperature, score_dtype)
        loss = F.cross_entropy(score.float(), torch.zeros(len(source), dtype=torch.long, device=device))
    if teacher is not None and teacher_weight:
        with torch.autocast(device.type, enabled=False):
            reference = teacher[torch.as_tensor(unique, device=device)]
            distance = 1 - (embedding.float() * reference.float()).sum(-1)
            # Count each source/target occurrence, so accumulation matches full batches.
            loss = loss + teacher_weight * distance[inverse].mean()
    return loss


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.float64)
    if len(ranks) == 0 or not np.all(np.isfinite(ranks)) or np.any(ranks < 1):
        raise ValueError("Invalid ranks")
    return {"mrr": float((1 / ranks).mean()), "hits1": float((ranks <= 1).mean()),
            "hits10": float((ranks <= 10).mean()), "queries": len(ranks)}


def average_ranks(scores, gold_columns):
    gold = scores[torch.arange(len(scores), device=scores.device), gold_columns]
    return 1 + (scores > gold[:, None]).sum(1).float() + ((scores == gold[:, None]).sum(1).float() - 1) / 2


def sample_negatives(positive, pool, count, rng, shared=False):
    if shared:
        eligible = pool[~np.isin(pool, positive)]
        if len(eligible) < count:
            raise ValueError("Insufficient negative pool")
        return np.broadcast_to(rng.choice(eligible, count, replace=False), (len(positive), count)).copy()
    result = np.empty((len(positive), count), dtype=np.int64)
    for i, value in enumerate(positive):
        eligible = pool[pool != value]
        result[i] = rng.choice(eligible, count, replace=False)
    return result


@torch.inference_mode()
def encode(model, events, ids, batch, amp):
    model.eval()
    result = []
    device = next(model.parameters()).device
    for start in range(0, len(ids), batch):
        rows = torch.as_tensor(events[ids[start:start + batch]], dtype=torch.long, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            central = torch.as_tensor(ids[start:start + batch], device=device)
            result.append(model(rows, central_ids=central))
    return torch.cat(result)


@torch.inference_mode()
def retrieve(query, target, pairs, candidate_ids, query_batch=128, target_batch=8192):
    """Exact average tie ranks without materializing the full query x candidate matrix."""
    lookup = {int(value): col for col, value in enumerate(candidate_ids)}
    gold = np.asarray([lookup[int(value)] for value in pairs[:, 1]], dtype=np.int64)
    ranks = []
    for start in range(0, len(query), query_batch):
        q = query[start:start + query_batch]
        columns = torch.as_tensor(gold[start:start + len(q)], device=q.device)
        # Obtain every gold score with the same matmul shape/precision as competitors.
        truth = torch.empty(len(q), device=q.device)
        for j in range(0, len(target), target_batch):
            selected = (columns >= j) & (columns < j + target_batch)
            if selected.any():
                block = q @ target[j:j + target_batch].T
                truth[selected] = block[selected, columns[selected] - j]
        greater = torch.zeros(len(q), device=q.device, dtype=torch.int64)
        equal = torch.zeros_like(greater)
        for j in range(0, len(target), target_batch):
            block = q @ target[j:j + target_batch].T
            if not torch.isfinite(block).all():
                raise FloatingPointError("Nonfinite retrieval scores")
            greater += (block > truth[:, None]).sum(1)
            equal += (block == truth[:, None]).sum(1)
        assert (equal >= 1).all(), "Gold score not found in exact retrieval"
        ranks.extend((1 + greater.float() + (equal.float() - 1) / 2).cpu().tolist())
    return np.asarray(ranks, dtype=np.float64)


@torch.inference_mode()
def retrieve_multiple(query, target, pairs, target_ids, candidate_sets, query_batch=128, target_batch=8192):
    """Reuse identical dot products across candidate subsets, including numerical ties."""
    lookup = {int(value): col for col, value in enumerate(target_ids)}
    subset_columns = {name: torch.as_tensor([lookup[int(e)] for e in ids], device=query.device)
                      for name, ids in candidate_sets.items()}
    subset_gold = {}
    for name, ids in candidate_sets.items():
        positions = {int(value): col for col, value in enumerate(ids)}
        subset_gold[name] = np.asarray([positions[int(e)] for e in pairs[:, 1]])
    result = {name: [] for name in candidate_sets}
    for start in range(0, len(query), query_batch):
        q = query[start:start + query_batch]
        score = torch.cat([q @ target[j:j + target_batch].T
                           for j in range(0, len(target), target_batch)], dim=1)
        if not torch.isfinite(score).all():
            raise FloatingPointError("Nonfinite retrieval scores")
        for name, columns in subset_columns.items():
            gold = torch.as_tensor(subset_gold[name][start:start + len(q)], device=query.device)
            result[name].extend(average_ranks(score[:, columns], gold).cpu().tolist())
    return {name: np.asarray(value, dtype=np.float64) for name, value in result.items()}


@torch.inference_mode()
def evaluate(model, events, pairs, candidates, args):
    tick = time.perf_counter()
    query = encode(model, events, pairs[:, 0], args.eval_batch, args.amp)
    target = encode(model, events, candidates, args.eval_batch, args.amp)
    ranks = retrieve(query, target, pairs, candidates, target_batch=args.score_block)
    measure = metrics(ranks)
    measure["seconds"] = time.perf_counter() - tick
    return measure, ranks


def run(args):
    output = Path(args.output)
    if (output / "config.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # The fused eval-only MHA path in some torch versions has mixed-dtype issues.
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
        torch.cuda.reset_peak_memory_stats()
    cache = Path(args.cache)
    protocol = json.loads((cache / "protocol.json").read_text(encoding="utf-8"))
    data = np.load(cache / "events.npz")
    events = data["events"]
    splits = np.load(cache / "splits.npz")
    train, dev = splits["train"], splits["dev"]
    candidates = splits[f"candidates_{args.candidates}"]
    config = vars(args).copy()
    config.update({"implementation": "independent paper reconstruction; topology disabled alpha=1",
                   "torch": torch.__version__, "numpy": np.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "negative_sampling": "16 shared distinct negatives per microbatch" if args.shared_negatives else "independent distinct negatives per query",
                   "selection": "validation MRR only; final test at selected checkpoint",
                   "protocol_sha256": sha256(cache / "protocol.json"),
                   "splits_sha256": sha256(cache / "splits.npz"),
                   "source_sha256": {str(p.name): sha256(p) for p in Path(__file__).parent.glob("*.py")}})
    if args.validation_only:
        config["selection"] = "validation MRR only; test evaluation disabled"
    if args.variant != "baseline":
        config["experimental_role"] = "controlled diagnostic variant; not the original architecture's reported result"
    if args.hard_negatives:
        config["negative_sampling"] = f"{args.hard_negatives} training-source temporal hard negatives plus {args.negatives - args.hard_negatives} random negatives"
    if args.teacher_weight:
        config["teacher"] = "fixed epoch-zero embeddings from the same observed graphs and training seeds; no validation/test labels"
    save_json(output / "config.json", config)
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        (source_dir / path.name).write_bytes(path.read_bytes())
    save_json(output / "protocol.json", protocol)
    from .controlled import build_model
    model = build_model(args, protocol, events, splits).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    target_pool = splits["candidates_all"]
    hard_pool = None
    if args.hard_negatives:
        from .controlled import temporal_hard_pool
        hard_pool = temporal_hard_pool(events, train, target_pool, protocol["time_tokens"])
        np.save(output / "training_hard_pool.npy", hard_pool)
    rng = np.random.default_rng(args.seed + 1)
    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({"start": str(output), "parameters": parameters, "config": config}), flush=True)
    best, best_epoch, stale = -1, 0, 0
    start_time = time.perf_counter()
    teacher = None
    if args.teacher_weight:
        teacher = encode(model, events, np.arange(len(events)), args.eval_batch, args.amp).detach().clone()
    with (output / "epochs.jsonl").open("w", encoding="utf-8", buffering=1) as log:
        if args.eval_initial:
            initial, ranks = evaluate(model, events, dev, candidates, args)
            best, best_epoch = initial["mrr"], 0
            torch.save({"model": model.state_dict(), "epoch": 0}, output / "best.pt")
            np.save(output / "best_dev_ranks.npy", ranks)
            save_json(output / "best_dev.json", {"epoch": 0, **initial})
            initial_record = {"epoch": 0, "initial": True, "dev": initial}
            print(json.dumps(initial_record), flush=True)
            log.write(json.dumps(initial_record) + "\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            tick, total_loss = time.perf_counter(), 0.0
            order = rng.permutation(len(train))
            for group_start in range(0, len(order), args.batch):
                group = order[group_start:group_start + args.batch]
                optimizer.zero_grad(set_to_none=True)
                for micro_start in range(0, len(group), args.microbatch):
                    selected = group[micro_start:micro_start + args.microbatch]
                    positive = train[selected, 1]
                    if hard_pool is None:
                        negatives = sample_negatives(positive, target_pool, args.negatives, rng, args.shared_negatives)
                    else:
                        from .controlled import mixed_negatives
                        negatives = mixed_negatives(positive, target_pool, hard_pool[selected], args.negatives,
                                                     args.hard_negatives, rng)
                    targets = np.column_stack([positive, negatives])
                    loss = sampled_loss(model, events, train[selected, 0], targets, device, args.temperature,
                                        args.amp, args.train_score_dtype, teacher, args.teacher_weight)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
                    (loss * len(selected) / len(group)).backward()
                    total_loss += loss.item() * len(selected)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            record = {"epoch": epoch, "loss": total_loss / len(train), "train_seconds": time.perf_counter() - tick}
            if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
                measure, ranks = evaluate(model, events, dev, candidates, args)
                record["dev"] = measure
                if measure["mrr"] > best + 1e-9:
                    best, best_epoch, stale = measure["mrr"], epoch, 0
                    torch.save({"model": model.state_dict(), "epoch": epoch}, output / "best.pt")
                    np.save(output / "best_dev_ranks.npy", ranks)
                    save_json(output / "best_dev.json", {"epoch": epoch, **measure})
                else:
                    stale += 1
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            save_json(output / "status.json", {"state": "training", "epoch": epoch, "best_epoch": best_epoch,
                       "best_dev_mrr": best, "elapsed_seconds": time.perf_counter() - start_time})
            if stale >= args.patience:
                break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if args.validation_only:
        measure, ranks = evaluate(model, events, dev, candidates, args)
        np.save(output / "validation_ranks.npy", ranks)
        final = {"best_epoch": best_epoch, "last_epoch": epoch, "validation": measure,
                 "seconds": time.perf_counter() - start_time, "parameters": parameters,
                 "optimizer_updates": epoch * int(np.ceil(len(train) / args.batch)),
                 "test_evaluated": False, "variant": args.variant,
                 "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None}
        save_json(output / "results_validation.json", final)
        save_json(output / "status.json", {"state": "completed_validation_only", **final})
        print(json.dumps({"completed": final}), flush=True)
        return
    # Final retrieval under three explicit candidate protocols; no further selection.
    test = splits["test"]
    final = {"best_epoch": best_epoch, "best_dev_mrr": best, "last_epoch": epoch, "test": {}}
    query = encode(model, events, test[:, 0], args.eval_batch, args.amp)
    target_ids = splits["candidates_all"]
    target = encode(model, events, target_ids, args.eval_batch, args.amp)
    candidate_sets = {name: splits[f"candidates_{name}"] for name in ("all", "nontrain", "reference")}
    ranked = retrieve_multiple(query, target, test, target_ids, candidate_sets, target_batch=args.score_block)
    for setting, ranks in ranked.items():
        ids = candidate_sets[setting]
        np.save(output / f"test_ranks_{setting}.npy", ranks)
        final["test"][setting] = {"candidates": len(ids), **metrics(ranks)}
    final.update({"seconds": time.perf_counter() - start_time, "parameters": parameters,
                  "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None})
    save_json(output / "results.json", final)
    save_json(output / "status.json", {"state": "completed", **final})
    print(json.dumps({"completed": final}), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--positions", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--negatives", type=int, default=16)
    p.add_argument("--shared-negatives", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--score-block", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--candidates", choices=["all", "nontrain", "reference"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-fraction", type=float, default=0.10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validation-only", action="store_true")
    from .controlled import VARIANTS
    p.add_argument("--variant", choices=VARIANTS, default="baseline")
    p.add_argument("--time-weight", type=float, default=0.8)
    p.add_argument("--hard-negatives", type=int, default=0)
    p.add_argument("--train-score-dtype", choices=["autocast", "float32"], default="float32")
    p.add_argument("--eval-initial", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--teacher-weight", type=float, default=0.0)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
