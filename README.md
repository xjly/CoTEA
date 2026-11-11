# MSEA-Align

## Run the experiments

Run the commands from the repository root. The training and full evaluations require an NVIDIA GPU with CUDA. Use Python 3.10 and PyTorch 2.6 with CUDA 11.8 to match the baseline environment. Select one GPU with `CUDA_VISIBLE_DEVICES`.

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r reproduction/requirements.txt numpy==1.26.4 scipy==1.15.2
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

### WUXI-TEMP

```bash
python experiments/wuxi_suite_20261108/prepare_and_baselines.py
python experiments/wuxi_suite_20261108/train_wuxi_suite.py --arm no_period
python experiments/wuxi_suite_20261108/train_wuxi_suite.py --arm hier_time
python experiments/wuxi_suite_20261108/test_wuxi_breakthrough.py
python experiments/wuxi_suite_20261108/test_wuxi_breakthrough.py --checkpoint-step 400 --no-sinkhorn
```

The calendar comparison uses `step_200/dual_full_notime_015` in each arm's `results_wuxi.json`. The main and ablation results are written to `synergy_results.json`.

### BETA-30%

```bash
python -m reproduction.prepare_beta_seed30 --raw datasets/BETA --output runs/beta30
python -m reproduction.beta_seed30_train --prepared runs/beta30 --output runs/beta30/step375.pt
python -m reproduction.beta30_readout --prepared runs/beta30 --checkpoint runs/beta30/step375.pt --output runs/beta30/results.json
```

### DICEWS-1K

```bash
python -m experiments.dicews prepare
python -m experiments.dicews train --gpu 0
python -m experiments.dicews evaluate --gpu 0
python -m experiments.dicews disjoint --gpu 0
```

### Comparison methods and protocol audits

The comparison runners additionally use PyTorch Geometric, torch-scatter, GPU Faiss, pandas, tqdm and easydict. For the environment above:

```bash
python -m pip install torch-geometric==2.6.1 pandas==2.2.3 tqdm==4.67.1 easydict faiss-gpu-cu11==1.10.0
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.6.0+cu118.html
python -m experiments.prepare_baselines

RCTEA_OUTPUT="$PWD/runs/dicews/rctea" PYTHONHASHSEED=0 \
  bash -c 'cd third_party/rctea/dicews && python main.py --random_seed 2026'

python experiments/wuxi_suite_20261108/prepare_and_baselines.py --lighttea
python -m experiments.wuxi_suite_20261108.run_four_baselines
python -m experiments.wuxi_suite_20261108.run_dicews_all_baselines
for method in light stea_htea mgtea rctea; do
  python -m experiments.wuxi_suite_20261108.run_beta30_all_baselines --task "$method"
done
python -m experiments.wuxi_suite_20261108.run_protocol_audits
```

Run the main experiments before the protocol audits. The comparison and audit scripts write their results under `experiments/wuxi_suite_20261108`.
