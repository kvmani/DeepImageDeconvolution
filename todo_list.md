# TODO List

This is the operational task list. Update it whenever priorities change, work is completed, or new gaps are discovered.

## Main Next Tasks (Priority)

- [x] Objective: update pipeline to predict mixing weights (x, y) alongside A and B.
- [x] Models: implement `src/models/` with Dual U-Net blocks and factory.
- [x] Training: add One-Cycle LR policy and refactor training into engine/optim/checkpoint/metrics modules.
- [x] Docs: add Dual U-Net architecture documentation and centralized IEEE references.
- [x] Reporting: emit report.json + monitoring figures and build Quarto summary decks.
- [ ] Data: build a larger synthetic dataset and record the exact config and version in docs.
- [ ] Training: run a baseline training on that dataset and capture key metrics plus monitoring output.
- [x] Evaluation: add a CLI evaluation report (PSNR/SSIM/L2) and a CI workflow for pytest + debug smoke runs.
- [x] Inference: ship a PySide6-based GUI with model caching, metrics export, and manifest logging.

## Now (Next 1-2 weeks)

- [x] Data: add recursive input scanning for nested datasets in generation/prep scripts.
- [x] Data: downscale repo demo BMPs to 8-bit for documentation and record the 16-bit rescaling requirement.
- [x] Reporting: update training report.json periodically (status/progress/tracking sample) and add a run-comparison helper script.
- [x] Reporting: write per-epoch metrics CSV and surface it in report.json artifacts.
- [x] Reporting: integrate run comparison table into summarize_results deck.
- [ ] Data: expand inputs using `data/raw/Double Pattern Data` and document preprocessing assumptions.
- [ ] Data: decide the default mixing pipeline and weight distribution for training, then sync configs/docs.
- [x] Training: standardize experiment naming and output folder conventions for reproducibility.
- [ ] Docs: add a short baseline-results section to `docs/training_inference.md`.
- [x] Reporting: add lab meeting demo deck with reproducible mixing artifacts and report.json.
- [x] Data: add a non-destructive preparation script for experimental BMPs with manifest checksums.
- [x] Metrics: add mask-aware reconstruction metrics and surface them in monitoring.
- [x] Workflow: add `--run-tag` to standardize timestamped output directories.
- [x] Docs: add a manuscript-style methods document with equations and assumptions.
- [x] Training: add CLI `--set` overrides with strict precedence and resolved config snapshots.
- [x] Workflow: add `run_train_jobs.sh` to batch sequential GPU training runs with logging.

## Next

- [ ] Validation: evaluate on real experimental double-pattern data with a clear preprocessing path.
- [ ] Models: prototype one additional architecture (GAN or attention variant) and document results.
- [ ] Metrics: add physics-informed or orientation-based evaluation metrics if feasible.
- [ ] Analysis: run an ablation study comparing mixing pipelines and weight ranges.

## Later

- [ ] Packaging: provide a reproducible environment file (Conda or Docker) with pinned dependencies.
- [ ] Performance: add mixed precision and data-loader tuning options.
- [ ] Release: create a release checklist and reproducibility bundle for external users.

## Recent Completions

- [x] Docs: add reproducibility, results, ablation protocol, and data provenance documents.
- [x] Evaluation: add `scripts/evaluate_real_data.py` for masked real-data metrics and qualitative grids.
- [x] QA: add CI (pytest + debug train smoke + report validation) and ruff/pre-commit tooling.
- [x] Tests: add report schema validation and compare_runs smoke test.
- [x] Inference: align GT preprocessing with input preprocessing for metrics and support mixed GT keys in GUI exports.
- [x] Docs: expand inference CLI and GUI usage guidance.
