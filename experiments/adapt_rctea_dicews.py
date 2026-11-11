"""Apply the DICEWS-1K adapter to a clean copy of the paper's RCTEA code."""

import shutil
import sys
from pathlib import Path


def replace(source, old, new):
    if old not in source:
        raise RuntimeError(f"Upstream source no longer matches: {old!r}")
    return source.replace(old, new)


upstream, output = map(Path, sys.argv[1:])
shutil.copytree(upstream, output, ignore=shutil.ignore_patterns("__pycache__"))
for filename in ("model_t.py", "model_o.py"):
    path = output / filename
    path.write_text(replace(path.read_text(encoding="utf-8"), "self.time_int_emb[2542]", "self.time_int_emb[self.zero_interval]"), encoding="utf-8")

path = output / "main.py"
source = path.read_text(encoding="utf-8")
source = replace(source, "import tensorflow as tf\n", "import random\nimport json\nfrom pathlib import Path\n")
source = replace(source, "from seu_tkg import sinkhorn, cal_sims\nfrom CSLS_test_4 import *\n", "")
source = replace(source, "filename = 'YAGO-WIKI50K/'", "filename = 'ICEWS05-15/'")
source = replace(source, "#set_seed(cfgs.random_seed)", """random.seed(cfgs.random_seed)
np.random.seed(cfgs.random_seed)
torch.manual_seed(cfgs.random_seed)
output_dir = Path(os.environ["RCTEA_OUTPUT"])
output_dir.mkdir(parents=True, exist_ok=True)
print(f"RUN seed={cfgs.random_seed}")""")
source = replace(source, "train_ratio=2000", "train_ratio=1000")
source = replace(source, "print(train_pair.shape, dev_pair.shape)", """print(train_pair.shape, dev_pair.shape)
assert train_pair.shape == (1000, 2) and dev_pair.shape == (7566, 2)
assert all(len(np.unique(np.concatenate([train_pair[:, i], dev_pair[:, i]]))) == 8566
           for i in range(2))
np.save(output_dir / "test_pairs.npy", dev_pair)""")
for model in ("model_time", "model_over"):
    old = f"{model} = {model}.to(device)"
    source = replace(source, old, f"{old}\n{model}.zero_interval = time_int_dict_i[(0, 0)]")
source = replace(source, "np.arange(2000, 187987, 1)", "np.arange(len(train_pair), N_s)")
# Use staged progress and fail early if any optimizer step becomes non-finite.
for loss in ("loss_ent", "loss_time", "loss"):
    source = replace(source, f"print({loss})", f"assert torch.isfinite({loss}).item(), 'Non-finite {loss}'")
source = source[:source.index("#the test, requires memory")]
source += '''
# Same normalization and 10 row/column passes as seu_tkg.py, using PyTorch.
model_over.eval()
model_over.return_loss = False
with torch.no_grad():
    features = model_over([adj_matrix, train_pair])
    features = features / (features.norm(dim=-1, keepdim=True) + 1e-5)
    left = features[dev_pair[:, 0]]
    right = features[dev_pair[:, 1]]
    score = (50 * (left @ right.T)).exp()
    for _ in range(10):
        score.div_(score.sum(dim=1, keepdim=True) + 1e-6)
        score.div_(score.sum(dim=0, keepdim=True) + 1e-6)
    assert torch.isfinite(score).all().item()
    # The candidate order matches the pair order, so truth is on the diagonal.
    ranks = torch.cat([
        (rows.argsort(descending=True, dim=1) ==
         torch.arange(start, start + len(rows), device=device)[:, None])
        .to(torch.int64).argmax(dim=1) + 1
        for start in range(0, len(score), 256)
        for rows in [score[start:start + 256]]
    ]).cpu().numpy()
np.save(output_dir / "ranks.npy", ranks)
np.save(output_dir / "embeddings.npy", features.cpu().numpy())
metrics = dict(seed=cfgs.random_seed, n=len(ranks), mrr=float(np.mean(1 / ranks)),
               hits1=float(np.mean(ranks <= 1)), hits10=float(np.mean(ranks <= 10)))
(output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\\n")
print("RESULT", json.dumps(metrics))
'''
path.write_text(source, encoding="utf-8")
