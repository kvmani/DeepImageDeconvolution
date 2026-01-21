# Reproducibility Guide

This guide provides exact commands for data preparation, training, inference, and reporting. It is the canonical path for reproducing results.

## Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Data preparation (experimental inputs)

The repo ships 8-bit demo copies under `data/raw/Double Pattern Data/`. Convert them to canonical 16-bit PNGs before processing:

```bash
python3 scripts/prepare_experimental_data.py \
  --input-dir "data/raw/Double Pattern Data" \
  --output-dir "data/processed/Double Pattern Data" \
  --output-format png \
  --output-bit-depth 16
```

## Synthetic data generation

```bash
python3 scripts/generate_data.py \
  --config configs/default.yaml \
  --input-dir "data/processed/Double Pattern Data" \
  --recursive-input \
  --output-dir "data/synthetic/baseline" \
  --num-samples 2000 \
  --seed 42
```

## Training (baseline)

```bash
python3 scripts/run_train.py \
  --config configs/train_default.yaml \
  --out_dir outputs/train_baseline
```

The training run updates `report.json` each epoch (status/progress + tracking sample paths) and writes `history.csv` alongside `history.json`.

## Inference

```bash
python3 scripts/run_infer.py \
  --config configs/infer_default.yaml \
  --checkpoint outputs/train_baseline/best.pt \
  --out_dir outputs/infer_baseline
```

## Real-data evaluation

```bash
python3 scripts/evaluate_real_data.py \
  --config configs/eval_real_default.yaml \
  --checkpoint outputs/train_baseline/best.pt \
  --out_dir outputs/eval_real_baseline
```

## Build the summary deck

```bash
python3 reports/summarize_results/scripts/build_report.py --run-id train_baseline
quarto render reports/summarize_results/deck.qmd --to pdf
```

## Compare runs

```bash
python3 scripts/compare_runs.py --glob "outputs/*/report.json" \
  --out reports/summarize_results/run_comparison.csv
```

## Sweep runs (grid search)

```bash
python3 scripts/sweep_runs.py \
  --config configs/train_default.yaml \
  --grid train.learning_rate=1e-4,2e-4 \
  --grid train.loss.lambda_recon=0.3,0.5 \
  --out-root outputs/sweeps \
  --run-tag ablation_01
```

Each sweep writes `outputs/sweeps/<run-tag>/runs_index.json` with per-run metadata and report paths.
