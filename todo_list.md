# TODO List

This is the operational task list. Update it whenever priorities change, work is completed, or new gaps are discovered.

## Main Next Tasks (Priority)

- [x] Objective: update pipeline to predict mixing weights (x, y) alongside A and B.
- [x] Models: implement `src/models/` with Dual U-Net blocks and factory.
- [x] Training: add One-Cycle LR policy and refactor training into engine/optim/checkpoint/metrics modules.
- [x] Docs: add Dual U-Net architecture documentation and centralized IEEE references.
- [ ] Data: build a larger synthetic dataset and record the exact config and version in docs.
- [ ] Training: run a baseline training on that dataset and capture key metrics plus monitoring output.
- [ ] Evaluation: add a CLI evaluation report (PSNR/SSIM/L2) and a CI workflow for pytest + debug smoke runs.

## Now (Next 1-2 weeks)

- [ ] Data: expand inputs using `data/raw/Double Pattern Data` and document preprocessing assumptions.
- [ ] Data: decide the default mixing pipeline and weight distribution for training, then sync configs/docs.
- [x] Training: standardize experiment naming and output folder conventions for reproducibility.
- [ ] Metrics: export per-epoch metrics to CSV in addition to `history.json`.
- [ ] Docs: add a short baseline-results section to `docs/training_inference.md`.
- [x] Data: add a non-destructive preparation script for experimental BMPs with manifest checksums.
- [x] Metrics: add mask-aware reconstruction metrics and surface them in monitoring.
- [x] Workflow: add `--run-tag` to standardize timestamped output directories.
- [x] Docs: add a manuscript-style methods document with equations and assumptions.

## Next

- [ ] Validation: evaluate on real experimental double-pattern data with a clear preprocessing path.
- [ ] Models: prototype one additional architecture (GAN or attention variant) and document results.
- [ ] Metrics: add physics-informed or orientation-based evaluation metrics if feasible.
- [ ] Analysis: run an ablation study comparing mixing pipelines and weight ranges.

## Later

- [ ] Packaging: provide a reproducible environment file (Conda or Docker) with pinned dependencies.
- [ ] Performance: add mixed precision and data-loader tuning options.
- [ ] Release: create a release checklist and reproducibility bundle for external users.
