# TODO List

This is the operational task list. Update it whenever priorities change, work is completed, or new gaps are discovered.

## Main Next Tasks (Priority)

- [ ] Data: build a larger synthetic dataset and record the exact config and version in docs.
- [ ] Training: run a baseline training on that dataset and capture key metrics plus monitoring output.
- [ ] Evaluation: add a CLI evaluation report (PSNR/SSIM/L2) and a CI workflow for pytest + debug smoke runs.

## Now (Next 1-2 weeks)

- [ ] Data: expand inputs using `data/raw/Double Pattern Data` and document preprocessing assumptions.
- [ ] Data: decide the default mixing pipeline and weight distribution for training, then sync configs/docs.
- [ ] Training: standardize experiment naming and output folder conventions for reproducibility.
- [ ] Metrics: export per-epoch metrics to CSV in addition to `history.json`.
- [ ] Docs: add a short baseline-results section to `docs/training_inference.md`.

## Next

- [ ] Validation: evaluate on real experimental double-pattern data with a clear preprocessing path.
- [ ] Models: prototype one additional architecture (GAN or attention variant) and document results.
- [ ] Metrics: add physics-informed or orientation-based evaluation metrics if feasible.
- [ ] Analysis: run an ablation study comparing mixing pipelines and weight ranges.

## Later

- [ ] Packaging: provide a reproducible environment file (Conda or Docker) with pinned dependencies.
- [ ] Performance: add mixed precision and data-loader tuning options.
- [ ] Release: create a release checklist and reproducibility bundle for external users.
