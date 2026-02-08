# Deterministic Inversion Usage

This document describes the implemented deterministic (non-ML) mixing-fraction inversion workflow.

## 1) Run commands

Default run (synthetic single-pair):

```bash
python3 scripts/invert_mixing_fraction.py
```

Debug run (synthetic single-pair, faster grid):

```bash
python3 scripts/invert_mixing_fraction.py --debug
```

With explicit config + run tag:

```bash
python3 scripts/invert_mixing_fraction.py \
  --config configs/deterministic_inversion/inversion_default.yaml \
  --run-tag baseline_detinv_01
```

Override the synthetic A/B inputs and `x_true`:

```bash
python3 scripts/invert_mixing_fraction.py \
  --set data.synthetic_pair.a_path="data/raw/Double Pattern Data/Good Pattern/Perfect_BCC-1.bmp" \
  --set data.synthetic_pair.b_path="data/raw/Double Pattern Data/Good Pattern/Perfect_FCC-1.bmp" \
  --set data.synthetic_pair.x_true=0.65 \
  --set deterministic_inversion.search.exhaustive_step=0.005
```

Evaluate on real mixed patterns (batch mode):

```bash
python3 scripts/invert_mixing_fraction.py \
  --config configs/deterministic_inversion/inversion_real_default.yaml
```

Synthetic robustness benchmark (metric ranking with known `x_true`):

```bash
python3 scripts/summarize_metric_robustness.py --debug
```

## 2) Implemented defaults

Synthetic single-pair mode (default):

- A/B inputs: explicit `data.synthetic_pair.a_path` / `data.synthetic_pair.b_path`
- Synthetic mix: `C = x_true * A + (1-x_true) * B`
- Per-metric optimization: for each enabled metric, estimate `x_hat` that optimizes that metric, then report all metrics at `C_hat(x_hat)`

Real mixed-pattern batch mode (via `inversion_real_default.yaml`):

- Mixed input set: `data.mixed_dir` (default `data/raw/Double Pattern Data/50-50 Double Pattern`)
- Candidate pool: `data.candidate_pool.root_dir` (default `data/raw/Double Pattern Data/Good Pattern`) with `bcc` as A-type and `fcc` as B-type
- Masking: centered circular mask (enabled by default)
- Background correction: subtractive default; divisive optional
- Metrics: `ncc`, `ssim`, `l2`, `l1`
- Search: grid search (config-controlled)
- Alignment: config-controlled (enabled by default in real batch configs)

## 3) Output artifacts

Each run writes to `output.out_dir` (default `outputs/deterministic_inversion`):

- `resolved_config.yaml`
- `manifest.json`
- `report.json` (status/summary)
- `results.jsonl`
- `summary_metrics.csv` (per-metric optimization summary in synthetic mode; aggregate summary in batch mode)
- `curves/*.csv` (score-vs-fraction curves)
- `synthetic_inputs/A.png`, `synthetic_inputs/B.png`, `synthetic_inputs/C.png` (synthetic mode)
- `reconstructions/*.png` (includes `C_hat__opt_<metric>.png` in synthetic mode)
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
