# Ablation Protocol

This protocol standardizes ablations so results are comparable and reproducible.

## Fixed baseline settings

- Dataset: `data/synthetic/baseline` (document exact generation config and seed).
- Model: `unet_dual` (base_channels=32, depth=4).
- Training: 50 epochs, batch size 8, One-Cycle scheduler.
- Masking: enabled by default.
- Logging: per-epoch `report.json`, `history.csv`, and image log tracking sample.

## Ablation axes

1. **Mixing pipeline**
   - `normalize_then_mix`
   - `mix_then_normalize`

2. **Weight distribution**
   - Uniform `U(0.2, 0.8)`
   - Beta (e.g., `beta(2,2)`; document params)
   - Fixed weights (e.g., 50/50, 90/10)

3. **Masking**
   - Mask enabled (default)
   - Mask disabled (for sensitivity)

4. **Normalization**
   - Smart inside-mask normalization (`smart: true`)
   - Global normalization (`smart: false`)

## Required reporting

For each ablation run:

- Save `report.json` + `history.csv`.
- Record exact config in `config_used.json`.
- Add run ID and key metrics to `docs/results.md`.
- Generate a summary deck (`reports/summarize_results`).

## Example sweep command

```bash
python3 scripts/sweep_runs.py \
  --config configs/train_default.yaml \
  --grid data.mix.pipeline=normalize_then_mix,mix_then_normalize \
  --grid data.mix.weight.min=0.2,0.5 \
  --grid data.mix.weight.max=0.8,0.5 \
  --run-tag ablation_mix_01
```
