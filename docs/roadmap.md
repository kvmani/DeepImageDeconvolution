# Project Roadmap: Mixed Kikuchi Pattern Deconvolution

This roadmap outlines the phases and deliverables needed to realise the mission described in `mission_statement.md`. It should serve as a living document: update it as the project evolves, but resist the temptation to skip phases. Each phase builds upon the previous ones to ensure a robust, reproducible solution.

## Current Status Snapshot

For the current implementation snapshot and near-term tasks, see `docs/status.md` and `todo_list.md`. This roadmap is the long-term view; it is not a day-to-day task list.

## Phase 0 – Literature Survey and Requirements (Weeks 1–2)

**Objectives:**

* Survey recent research on EBSD pattern analysis, source separation, image deconvolution and deep learning architectures such as U‑Net, GANs and attention networks.  Summarise key findings, advantages and limitations.
* Identify state‑of‑the‑art open source implementations (PyTorch, TensorFlow) that could be adapted.
* Define success metrics (PSNR, SSIM, orientation difference) and realistic performance targets.

**Deliverables:**

* `docs/references/literature_survey.md` summarising important papers and key takeaways with citations.
* A requirements document specifying data bit‑depth, target image resolution, acceptable processing latency and any regulatory constraints.

## Phase 1 – Data Acquisition and Pre‑Processing (Weeks 3–5)

**Objectives:**

* Collect a diverse set of high quality pure Kikuchi patterns.  This may involve experimental EBSD measurements or synthetically generated patterns using diffraction simulation software.
* Implement pre‑processing routines in `src/preprocessing/` to verify 16‑bit input, crop regions of interest, apply circular masks, normalise intensities (including smart normalization inside the mask), and optionally denoise.  Scripts should be configurable and run in debug mode on a sample dataset.

**Deliverables:**

* `data/raw/` populated with pure pattern images (organised by phase or orientation).
* `src/preprocessing/normalise.py` and associated unit tests.
* A report describing pre‑processing choices and their impact on pattern quality.

## Phase 2 – Synthetic Data Generation (Weeks 6–8)

**Objectives:**

* Design and implement synthetic mixing utilities under `src/generation/` according to the strategies outlined in the mission statement: normalise‑then‑mix and mix‑then‑normalise.  Include optional noise addition and point spread convolution.
* Generate a large paired dataset \((C,(A,B))\) stored in `data/synthetic/`.  Provide a script `scripts/generate_data.py` with a `--debug` mode that creates a small sample dataset for testing.

**Deliverables:**

* `data/synthetic/` containing thousands of mixed and ground truth pattern triplets.
* `src/generation/mix.py` with extensive unit tests covering corner cases (e.g., extreme weight values, mismatched shapes).
* Documentation describing mixing strategies and their physical justification.

## Phase 3 – Model Architecture Design (Weeks 9–11)

**Objectives:**

* Evaluate candidate architectures (U‑Net, dual‑output U‑Net, conditional GANs with cycle consistency, attention enhancements).  Prototype each in `src/models/` with configurable depth, width and attention modules.
* Select a baseline model based on literature survey and preliminary experiments.

**Deliverables:**

* Implementation files such as `src/models/unet_dual.py`, `src/models/gan_dual.py` with clear docstrings and tests.
* A comparative report summarising the pros and cons of each architecture and justifying the choice of baseline.

## Phase 4 – Training Pipeline Development (Weeks 12–15)

**Objectives:**

* Build the training loop in `src/training/train.py`: handle data loading, batching, model instantiation, loss calculation, optimisation, checkpointing and logging.  Support both CPU and GPU.
* Ensure the pipeline respects debug mode: quick runs on small datasets with verbose output.
* Integrate configuration management via YAML (`configs/default.yaml`, `configs/debug.yaml`).

**Deliverables:**

* `run_train.py` CLI wrapper that accepts a config file and runs training.
* Training curves, logs and checkpoints saved under `models/` for the baseline model on the synthetic dataset.
* Documentation explaining how to configure and run training on various platforms.

## Phase 5 – Inference and Post‑Processing (Weeks 16–17)

**Objectives:**

* Develop inference scripts in `src/inference/infer.py` capable of loading a trained model and applying it to new mixed patterns.  The script should output deconvoluted 16‑bit images and optionally overlay difference maps.
* Implement post‑processing to ensure that the predicted \(\hat{A}\), \(\hat{B}\), and \(\hat{x}\) reconstruct the input \(C\) via \(\hat{C} = \hat{x}\hat{A} + (1-\hat{x})\hat{B}\) and that intensities remain within bounds.

**Deliverables:**

* `run_infer.py` CLI wrapper for inference.
* Sample outputs on held‑out test data with quantitative metrics.
* Scripts for visualising predictions alongside ground truth and the mixed input.

## Phase 6 – Evaluation and Metrics (Weeks 18–19)

**Objectives:**

* Formalise evaluation metrics (PSNR, SSIM, orientation difference) in `src/utils/evaluation.py`.  Provide functions that accept predicted and ground truth images and return metric scores.
* Benchmark the baseline model on a test set.  Compare against simpler baselines (e.g., direct intensity splitting) to quantify improvement.

**Deliverables:**

* A report summarising quantitative results and qualitative observations.  Highlight failure modes and potential improvements.
* Unit tests for metric functions.

## Phase 7 – Integration and Packaging (Weeks 20–21)

**Objectives:**

* Organise the repository for release.  Provide a `README.md` with installation instructions, usage examples, and citations of the mission statement and roadmap.
* Create a Dockerfile or Conda environment file that reproduces the environment on Windows and Linux.  Ensure all dependencies are pinned and no external network calls occur during inference.
* Optional: wrap the deconvolution pipeline into a simple CLI or API suitable for integration into larger EBSD analysis workflows.

**Deliverables:**

* A packaged release candidate with all code, tests and documentation.  A zipped dataset of synthetic patterns may accompany the release if licensing permits.
* Environment definition files (`environment.yml`, `requirements.txt`) tested on multiple platforms.

## Phase 8 – Documentation and Dissemination (Weeks 22–23)

**Objectives:**

* Finalise `mission_statement.md`, `agents.md`, and `roadmap.md` to reflect the implemented solution.  Cross‑link relevant sections.
* Write user‑level documentation explaining how to prepare data, train models, and perform inference.  Include FAQs and troubleshooting guides.
* Prepare a presentation or publication summarising the project for the materials science community.

**Deliverables:**

* Up‑to‑date documentation in the `docs/` folder.
* A slide deck or report ready for dissemination.

## Phase 9 – Maintenance and Future Work (Week 24+)

After the initial release, future work may include:

* Extending the model to more than two mixed patterns.
* Incorporating physics‑based constraints or generative priors into the network.
* Running ablation studies on mixing strategies and model components.
* Optimising for inference speed and memory footprint for deployment on resource‑constrained systems.

Such tasks should be added to this roadmap as separate phases with their own objectives and deliverables.

---

By following this roadmap, the team can develop a scientifically rigorous and production‑ready solution to the mixed Kikuchi deconvolution problem.  Each phase is intentionally scoped to build foundations before tackling more complex tasks, ensuring that the final system is reliable, interpretable and maintainable.
