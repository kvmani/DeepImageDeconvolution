# Deterministic Inversion Usage

This document describes the implemented deterministic (non-ML) mixing-fraction inversion workflow.

## 1) Run commands

Default run:

```bash
python3 scripts/invert_mixing_fraction.py
```

Debug run (fast, deterministic):

```bash
python3 scripts/invert_mixing_fraction.py --debug
```

With explicit config + run tag:

```bash
python3 scripts/invert_mixing_fraction.py \
  --config configs/deterministic_inversion/inversion_default.yaml \
  --run-tag baseline_detinv_01
```

Common overrides:

```bash
python3 scripts/invert_mixing_fraction.py \
  --mixed-dir "data/raw/Double Pattern Data/50-50 Double Pattern" \
  --candidate-root "data/raw/Double Pattern Data/Good Pattern" \
  --a-pattern "(?i)bcc" \
  --b-pattern "(?i)fcc" \
  --set deterministic_inversion.search.grid_steps='[0.1,0.02,0.005]'
```

Synthetic robustness benchmark (metric ranking with known `x_true`):

```bash
python3 scripts/summarize_metric_robustness.py --debug
```

## 2) Implemented defaults

- Mixed input set: `data/raw/Double Pattern Data/50-50 Double Pattern`
- Candidate pool: `data/raw/Double Pattern Data/Good Pattern` with `bcc` as A-type and `fcc` as B-type
- Masking: centered circular mask (enabled by default)
- Background correction: subtractive default; divisive optional
- Metrics: `ncc` (primary), `ssim`, `l2`, `l1`
- Search: coarse-to-fine grid by default
- Alignment: enabled by default
  - translation enabled (default max shift 15 px; debug config uses 5 px)
  - rotation enabled (default ±2° search, hard maximum ±5°, step 0.5°)
  - interpolation order 3 (bicubic)

## 3) Output artifacts

Each run writes to `output.out_dir` (default `outputs/deterministic_inversion`):

- `resolved_config.yaml`
- `manifest.json`
- `report.json` (status/progress summary; updated during run)
- `results.jsonl` (one record per mixed sample)
- `summary_metrics.csv`
- `curves/*.csv` (score-vs-fraction curves for best pair per sample)
- `reconstructions/*_C_hat.png` (16-bit reconstructed mixed pattern)
- `monitoring/qualitative/*.png` (qualitative panels)
- `report/index.html` (if enabled)

If failures occur, an `error_report.json` is also written.

Robustness benchmark runs (default `outputs/deterministic_robustness`) additionally write:

- `benchmark_results.jsonl` (per-sample `x_true`, nuisance settings, `x_hat` per metric)
- `summary_metrics.csv` (metric ranking by MAE/RMSE/bias/std)
- `summary_plots/mae_by_metric.png`
- `synthetic_samples/` (saved 16-bit A/B/C synthetic examples, limited by config)

## 4) Library-first policy

Before implementing new pattern-processing logic, consult and prefer existing EBSD tooling where feasible:

- `diffpy`
- `kikuchipy`
- `pyebsdindex`
- `hyperspy`

This repository’s deterministic implementation is intentionally modular so those libraries can be integrated directly in subsequent iterations.

## 5) Robustness benchmark methodology (implemented)

The synthetic benchmark:

1. Selects A/B candidates from the configured pool.
2. Synthesizes mixed patterns using configured `x_values`.
3. Applies configurable nuisance sweeps (gain/offset/noise/blur/shift/rotation).
4. Runs deterministic inversion for each synthetic sample.
5. Computes per-metric `|x_hat - x_true|`, then ranks metrics in `summary_metrics.csv`.

The benchmark defaults to `use_true_pair_only: true` so metric ranking isolates mixing-fraction recovery quality from candidate-selection ambiguity.
