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

Override the synthetic A/B inputs and `x_true` (scalar or list):

```bash
python3 scripts/invert_mixing_fraction.py \
  --set data.synthetic_pair.a_path="data/raw/Double Pattern Data/Good Pattern/Perfect_BCC-1.bmp" \
  --set data.synthetic_pair.b_path="data/raw/Double Pattern Data/Good Pattern/Perfect_FCC-1.bmp" \
  --set data.synthetic_pair.x_true='[0.1,0.3,0.5]' \
  --set deterministic_inversion.search.exhaustive_step=0.005
```

Add input noise/rotation to A/B for inversion:

```bash
python3 scripts/invert_mixing_fraction.py \
  --set data.synthetic_pair.noise.enabled=true \
  --set data.synthetic_pair.noise.gaussian_std=0.01 \
  --set data.synthetic_pair.noise.rotation_deg_max=2.0
```

Evaluate on real mixed patterns (batch mode):

```bash
python3 scripts/invert_mixing_fraction.py \
  --config configs/deterministic_inversion/inversion_real_default.yaml
```

Candidate-pool synthetic search (unknown A/B, random trials):

```bash
python3 scripts/invert_mixing_fraction.py \
  --config configs/deterministic_inversion/inversion_candidate_pool_default.yaml
```

Override candidate pool + trial count:

```bash
python3 scripts/invert_mixing_fraction.py \
  --config configs/deterministic_inversion/inversion_candidate_pool_default.yaml \
  --set data.candidate_pool.root_dir="data/raw/Double Pattern Data/Good Pattern" \
  --set data.candidate_pool.max_candidates=12 \
  --set data.candidate_pool.synthetic_pairs=3 \
  --set data.candidate_pool.x_true='[0.2,0.5,0.8]'
```

Enable optional FFT band-pass filtering:

```bash
python3 scripts/invert_mixing_fraction.py \
  --set deterministic_inversion.preprocess.fft_filter.enabled=true \
  --set deterministic_inversion.preprocess.fft_filter.low_cut=0.05 \
  --set deterministic_inversion.preprocess.fft_filter.high_cut=0.7
```

Synthetic robustness benchmark (metric ranking with known `x_true`):

```bash
python3 scripts/summarize_metric_robustness.py --debug
```

Synthetic pair demo (sample candidates, create noisy `C`, then recover pair + `x_hat`):

```bash
python3 scripts/run_deterministic_pair_demo.py \
  --index-a 0 --index-b 1 --x 0.5 \
  --candidate-count 10 --sample-seed 7 \
  --gaussian-enabled --rotation-enabled \
  --run-tag lab_demo
```

Identify pair for an arbitrary mixed input `C`:

```bash
python3 scripts/identify_candidate_pair.py \
  --mixed-path data/raw/Double\ Pattern\ Data/50-50\ Double\ Pattern/<image>.bmp \
  --candidate-dir data/raw/Double\ Pattern\ Data/Good\ Pattern \
  --candidate-count 10 \
  --sample-seed 7 \
  --run-tag identify_demo
```

Launch the deterministic GUI wrapper:

```bash
python3 scripts/run_deterministic_pair_gui.py
```

## 2) Implemented defaults

Synthetic single-pair mode (default):

- A/B inputs: explicit `data.synthetic_pair.a_path` / `data.synthetic_pair.b_path`
- Synthetic mix: `C = x_true * A + (1-x_true) * B`
- Synthetic noise: `data.synthetic_pair.noise` (Gaussian + small rotation applied to A/B for inversion only)
- Per-metric optimization: for each enabled metric, estimate `x_hat` that optimizes that metric, then report all metrics at `C_hat(x_hat)`
- Optional FFT filter: `deterministic_inversion.preprocess.fft_filter` (radial band-pass, `low_cut`/`high_cut` in [0,1] of Nyquist radius)

Real mixed-pattern batch mode (via `inversion_real_default.yaml`):

- Mixed input set: `data.mixed_dir` (default `data/raw/Double Pattern Data/50-50 Double Pattern`)
- Candidate pool: `data.candidate_pool.root_dir` (default `data/raw/Double Pattern Data/Good Pattern`) with `bcc` as A-type and `fcc` as B-type
- Masking: centered circular mask (enabled by default)
- Background correction: subtractive default; divisive optional
- Metrics: `ncc`, `ssim`, `l2`, `l1`
- Search: grid search (config-controlled)
- Alignment: config-controlled (enabled by default in real batch configs)

Candidate-pool synthetic search (via `inversion_candidate_pool_default.yaml`):

- Candidate pool: `data.candidate_pool.root_dir` (all patterns in the directory)
- Random sampling: `data.candidate_pool.max_candidates` + `sample_seed`
- Random trials: `data.candidate_pool.synthetic_pairs` pairs, sampled without replacement
- Mixing fraction: `data.candidate_pool.x_true` (scalar or list; repeated/cycled to match trial count)
- Noisy templates: `data.candidate_pool.noise` (Gaussian + small rotation applied to templates only)
- Alignment: **off by default** (enable if needed)

Deterministic GUI/CLI demo defaults:

- `configs/deterministic_inversion/pair_demo_default.yaml`
- `configs/deterministic_inversion/pair_demo_debug.yaml`
- NCC primary, L2 tie-break
- two-stage search enabled by default
- centered circular mask enabled by default

## 3) Output artifacts

Each run writes to `output.out_dir` (default `outputs/deterministic_inversion`):

- `resolved_config.yaml`
- `manifest.json`
- `report.json` (status/summary)
- `results.jsonl`
- `summary_metrics.csv` (per-metric optimization summary in synthetic mode; aggregate summary in batch mode; per-trial summary in candidate-pool mode)
- `curves/*.csv` (score-vs-fraction curves)
- `synthetic_inputs/A.png`, `synthetic_inputs/B.png`, `synthetic_inputs/A_noisy.png`, `synthetic_inputs/B_noisy.png`, `synthetic_inputs/C_x####.png` (synthetic mode)
- `reconstructions/*.png` (includes `C_hat__opt_<metric>__x####.png` in synthetic mode)
- `monitoring/qualitative/*.png` (qualitative panels)
- `report/index.html` (if enabled)
- `candidate_pool.csv` (candidate pool manifest in candidate-pool synthetic mode)
- `candidate_trials/*` (per-trial A/B/C/C_hat artifacts in candidate-pool synthetic mode)

If failures occur, an `error_report.json` is also written.

GUI run outputs (default base `outputs/gui_pair_demo`) include the same machine-readable files plus
`synthetic/` and `reconstructions/` image folders and `report/index.html`.

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
