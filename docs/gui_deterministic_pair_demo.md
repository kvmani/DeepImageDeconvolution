# Deterministic Pair Demo GUI

This document describes the PySide6 GUI for deterministic candidate-pair identification near grain boundaries.

## 1) Goal

The GUI is a wrapper around deterministic backend workflows:

1. Load a random candidate bank from `data/raw/Double Pattern Data/Good Pattern`.
2. Choose synthetic `A`, `B`, and `x` to generate `C = xA + (1-x)B`.
3. Add controlled nuisance to `A` and `B` (Gaussian, salt-pepper, rotation in `[-2°, +2°]`).
4. Recover the most likely pair and `x_hat` with NCC (primary) and L2 (tie-break).
5. Save logs, run manifests, and machine-readable results.

## 2) Launch

```bash
python3 scripts/run_deterministic_pair_gui.py
```

Optional:

```bash
python3 scripts/run_deterministic_pair_gui.py \
  --config configs/deterministic_inversion/pair_demo_default.yaml \
  --log-level INFO
```

## 3) GUI workflow

1. Set candidate folder and candidate count (`n=10` default).
2. Keep `Use fixed seed` enabled for reproducible sampling/noise.
3. Keep `Lock sampled candidates` enabled to reuse the same candidate strip.
4. Select `A index`, `B index`, and `x`.
5. Configure noise controls.
6. Click `Generate Synthetic C`.
7. Click `Identify Pair` (or `Run Full Demo`).

The bottom log panel streams stage-level progress and ETA:

- Stage-1 coarse candidate screening
- Stage-2 refinement over top-M pairs
- final winner summary and run directory

## 4) Search protocol (default)

The GUI uses a two-stage pair search:

1. **Coarse stage**: fast NCC screening over all unordered pairs.
2. **Refine stage**: full inversion/alignment on top-M pairs.

Default ranking:

- primary score: `NCC`
- tie-break: `L2` (lower is better)

`AB` and `BA` are treated as the same pair in search; the GUI reports `x_hat` in winner order and also reports reordered `x_hat` for the true synthetic order.

## 5) Output artifacts

Each run writes under:

`outputs/gui_pair_demo/<timestamp>_<run_tag>/`

Artifacts:

- `candidate_manifest.json`
- `demo_result.json`
- `top_k_pairs.csv`
- `manifest.json`
- `report.json`
- `report/index.html`
- `synthetic/*.png` (`A_noisy`, `B_noisy`, `C_synthetic`)
- `reconstructions/*.png` (`winner_a`, `winner_b`, `winner_c_hat`, `winner_residual_abs`)

## 6) Related CLI tools

- Generate one synthetic case:
  - `python3 scripts/create_synthetic_pair_case.py --index-a ... --index-b ... --x ...`
- Identify pair for any `C`:
  - `python3 scripts/identify_candidate_pair.py --mixed-path ...`
- Run complete synthetic demo in one command:
  - `python3 scripts/run_deterministic_pair_demo.py --index-a ... --index-b ... --x ...`
