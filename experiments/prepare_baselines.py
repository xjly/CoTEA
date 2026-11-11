"""Fetch the fixed upstream comparator implementations and inputs."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
RCTEA = (
    "https://raw.githubusercontent.com/psych-mind/RCTEA_/"
    "531c506f8c006cda62296539c4b64e0b974ee8d2/data_code/RCTEA"
)
MGTEA = (
    "https://raw.githubusercontent.com/DexterZeng/BETA/"
    "2092656055213cedde6cdaa275cc6f2c1ae6520d"
)
RCTEA_CODE = (
    "config_.py", "dgl_dual.py", "main.py", "model_o.py", "model_r.py",
    "model_t.py", "utils_n.py", "models/__init__.py", "models/dgmc.py",
    "models/gin.py", "models/mlp.py", "models/rel.py", "models/spline.py",
)


def fetch(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urlopen(url, timeout=90) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_beta_years(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for graph in (1, 2):
        raw = ROOT / "datasets/BETA" / f"triples_{graph}"
        target = destination / f"triples_{graph}c"
        with raw.open(encoding="utf-8") as source, target.open("w", encoding="utf-8", newline="\n") as output:
            for row in source:
                head, relation, tail, start, end = row.split()
                # The original comparator converts each date using its first four characters.
                years = ["0" if date == "~" else date[:4] for date in (start, end)]
                output.write("\t".join([head, relation, tail, *years]) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicews-data", type=Path, default=ROOT / "datasets/ICEWS05-15")
    args = parser.parse_args()
    data_files = ("triples_1", "triples_2", "sup_pairs", "ref_pairs")
    for name in data_files:
        if not (args.dicews_data / name).is_file():
            parser.error(f"Missing DICEWS input {args.dicews_data / name}.")

    third_party = ROOT / "third_party"
    rctea = third_party / "rctea"
    rctea.mkdir(parents=True, exist_ok=True)
    prepare_beta_years(rctea / "beta")
    fetch(f"{MGTEA}/models.py", third_party / "mgtea/MGTEA_models.py")
    fetch(f"{MGTEA}/LICENSE", third_party / "mgtea/LICENSE")

    adapted = rctea / "dicews"
    if not adapted.exists():
        with TemporaryDirectory(dir=rctea) as temporary:
            upstream = Path(temporary) / "upstream"
            for name in RCTEA_CODE:
                fetch(f"{RCTEA}/{name}", upstream / name)
            built = Path(temporary) / "adapted"
            subprocess.run(
                [sys.executable, str(ROOT / "experiments/adapt_rctea_dicews.py"), str(upstream), str(built)],
                check=True,
            )
            built.replace(adapted)
    target = adapted / "dataset/ICEWS05-15"
    target.mkdir(parents=True, exist_ok=True)
    for name in data_files:
        shutil.copyfile(args.dicews_data / name, target / name)


if __name__ == "__main__":
    main()
