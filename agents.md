# AGENTS.md – Guidelines for Developers and Automation Agents

This document describes how human developers and automated agents should work within this repository.  Its purpose is to ensure that all contributions align with the mission to deconvolve mixed Kikuchi patterns, respect the **16‑bit nature** of the data, and maintain high standards of code quality, reproducibility and transparency.  Where trade‑offs arise, **clarity and correctness** take precedence over unnecessary cleverness.  If you are unsure about a decision, document the rationale and err on the side of explicitness. Always consult docs/mission_statement.md for any clarificaiton of overall objectives and ensure the individual tasks and code changes are well aligned with the overall objecived defined there.

If an indivudual folder has another agents.md use that for fine tuning the expected behavior and give preference to its instruction where conflict arises with current document.

## 1. General Mandate

* **16‑bit Discipline**: All image handling routines **must operate on 16‑bit data**.  Never implicitly cast to 8‑bit unless explicitly for visualisation.  Input validation should assert bit‑depth and raise descriptive errors if expectations are violated.
* **Circular Masking**: Assume the central circular detector region is the meaningful signal.  Detect if inputs are already masked and, if not, apply the maximum inscribed circular mask (centered in the image) so outside pixels are zero.
* **Modular Design**: Structure the codebase into independent, testable modules.  Data loading, preprocessing, synthetic mixing, model definitions, training loops, and evaluation live in separate packages under `src/`.
* **Debug vs Regular Modes**: Every runnable script (data generation, training, inference) must support a `--debug` flag.  In debug mode the script should:
  - Use deterministic seeds.
  - Restrict itself to a small set of sample images or generate synthetic dummy patterns automatically.
  - Log verbosely and perform additional internal checks (e.g., verifying sum of outputs equals input).
  - When visualization is enabled, annotate the circular mask status and show the mask boundary in plots.
  - Finish quickly (<1 min), enabling rapid iteration.
  Regular mode uses full datasets and configured hyperparameters.
* **Cross‑Platform Support**: The code must run on Windows and Linux with CPU or GPU.  Use `os.path` for paths, avoid shell‑specific constructs, and rely on PyTorch device abstractions.  Do not hard‑code absolute directories; use configuration files and environment variables instead.
* **No Hidden Hard‑Coding**: All hyperparameters, file paths, and model sizes must be configurable via YAML or command‑line flags.  Single letters or magic numbers in code should either be removed or explained clearly.
* **Logging and Error Handling**: Use a central logging module (`src/utils/logging.py`) to obtain loggers.  Log at appropriate levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`) and include context (e.g., file names, shapes, bit‑depths).  Catch exceptions gracefully and present actionable messages.
* **Documentation Sync (Critical)**: Any change to code/scripts/configs that affects behavior, outputs, CLI flags, file formats, or assumptions **must** be reflected in documentation in the same change. Documentation must cover (1) usage/how-to-run, (2) algorithmic details, and (3) scientific assumptions/rationale in appropriate `.md` files. Create dedicated docs where helpful (e.g., under `docs/` or next to scripts) and ensure the top-level `README.md` links to them for discoverability.
* **Todo List Maintenance (Critical)**: Keep `todo_list.md` updated and current. Add, complete, or reprioritize items whenever behavior or priorities change so the list reflects the real state of the project.

## 2. Repository Structure (Authoritative)

```
kikuchi_deconvolution/
├─ docs/                    # Project documentation
│  ├─ mission_statement.md
│  ├─ agents.md
│  ├─ roadmap.md
│  └─ references/
├─ data/
│  ├─ raw/                 # Pure pattern images (16‑bit)
│  ├─ processed/           # Aligned and normalised images
│  └─ synthetic/           # Mixed patterns and ground truth pairs
├─ src/
│  ├─ datasets/            # Data loaders and dataset classes
│  ├─ preprocessing/       # Image normalisation, denoising, augmentation
│  ├─ generation/          # Synthetic mixing utilities
│  ├─ models/              # Neural network architectures (U‑Net, GANs, etc.)
│  ├─ training/            # Training loops, optimisers, callbacks
│  ├─ inference/           # Inference scripts and wrappers
│  ├─ utils/               # Logging, configuration, evaluation metrics
│  └─ __init__.py
├─ scripts/                # CLI wrappers (run_train.py, run_infer.py, generate_data.py)
├─ tests/                  # Unit and integration tests
├─ configs/                # YAML files with hyperparameters (debug.yaml, default.yaml)
├─ .gitignore
├─ README.md               # High level overview and quickstart
└─ AGENTS.md               # This file
```

### Notes on Structure

* Each package under `src/` should be importable without side effects.  Models should not load data, and data loaders should not instantiate models.
* `scripts/` contains thin wrappers that parse command‑line arguments, load configuration, instantiate modules from `src/`, and orchestrate the workflow.  They should remain minimal and free of algorithmic complexity.
* Tests under `tests/` follow the same structure as `src/`.  Use fixtures to create temporary 16‑bit images for testing.  Debug mode tests can reuse the `--debug` logic.

## 3. Coding Conventions

* **Language and Libraries**: Use Python 3.10+ and PyTorch (tested on CPU and CUDA). Ensure that code does work with python 3.11, and 3.12 seamlessly. Avoid experimental libraries unless justified and stable.
* **Type Hints**: All public functions and class methods must include type annotations.  Use `typing.Tuple`, `typing.List`, etc., for clarity.
* **Docstrings**: Follow the [NumPy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html) for functions, classes, and methods.  Describe the purpose, parameters, returns, raises, and examples if helpful.
* **PEP 8 Compliance**: Adhere to PEP 8 coding style.  Use `ruff` or `flake8` in CI to enforce formatting.  Long lines should be wrapped for readability.
* **Configuration**: Use YAML files in `configs/` for default hyperparameters.  Provide at least `default.yaml` for full training and `debug.yaml` for quick runs.  Configurations are loaded by `src/utils/config.py` and overrideable via CLI flags.
* **Logging**: All scripts and modules must call `get_logger(__name__)` from `src/utils/logging.py`.  Do not use `print()` except in notebooks or when absolutely necessary in debug mode.

## 4. Data Handling Guidelines

1. **Bit‑Depth Verification**: When loading an image, check `dtype` and `max()` values to confirm 16‑bit data.  If an 8‑bit image is detected, raise `ValueError` and include guidance on converting to 16‑bit.
2. **Circular Masking**: Apply the maximum inscribed circular mask after any cropping/augmentation.  Detect already-masked inputs (outside region ≈ 0) and record this in metadata/logs.
3. **Normalisation**: Convert pixel intensities to `float32` in the range \([0,1]\) using `astype(np.float32) / 65535.0`.  For masked images, compute normalization statistics **inside the circular mask** (smart normalization) so outside zeros do not skew the scale.  This smart normalization is project-specific and should remain configurable (default on).  Save any rescaling parameters for inverse transforms if necessary.
4. **Augmentations**: Provide augmentation utilities (flip, rotate, random crop) that preserve bit‑depth.  Augmentations should be parameterised and configurable in YAML.
5. **Synthetic Mixing**: Functions in `src/generation/` must implement the pipelines described in the mission statement (normalise‑then‑mix and mix‑then‑normalise).  They should accept two `numpy.ndarray` inputs, weights, and return a tuple `(C, (A, B))`.  Reapply the circular mask after any blur/noise so outside pixels remain zero.  For debug, allow deterministic weights and small images.
6. **Real Experimental Data**: When example, validation, or benchmark data is needed, use the dataset in `data/raw/Double Pattern Data/` and reference its README. Scripts, notebooks, and tests should use this data where appropriate, without modifying the raw files.

## 5. Model Development Guidelines

* Architectures must be implemented in separate files under `src/models/`.  Each model exposes a class with a `forward()` method taking a tensor of shape `(B, 1, H, W)` and returning either a concatenated tensor `(B, 2, H, W)` or a tuple `(A_pred, B_pred)`.
* Use `torch.nn.Module` subclasses and avoid global variables.  Parameter counts should be printed in debug mode.
* Weight initialisation should be explicit (e.g., Kaiming or Xavier) and configurable.
* Provide a `build_model(config: Dict) -> torch.nn.Module` factory function so that models are selected based on configuration.

## 6. Training and Inference Scripts

* `run_train.py` must parse arguments for config file location, debug flag, output directory, and random seed.  It initialises the dataset, model, optimiser, and learning rate scheduler, then runs training loops with periodic validation.
* `run_infer.py` should load a saved checkpoint and apply the model to new mixed patterns.  It must handle both single images and batches of images, saving deconvoluted outputs in 16‑bit format.
* Both scripts must log configuration and environment (OS, Python version, CUDA availability).  When exceptions occur, log the stack trace and exit gracefully.

## 7. Testing and Continuous Integration

* Write unit tests for every module in `src/`.  Tests must include 16‑bit edge cases, such as maximum intensity and zero intensity patterns, and ensure that synthetic mixing functions obey the sum rule \(\hat{A} + \hat{B} = C\).
* Provide integration tests that run `run_train.py --debug` and `run_infer.py --debug` to ensure the entire pipeline works end‑to‑end on a small dataset.


## 8. Contribution and Review Process

1. **Planning**: Before writing code, define the scope of the activity.  Identify corner cases (e.g., missing values, mismatched image shapes, invalid bit‑depths) and outline the algorithm in comments or a design doc.
2. **Implementation**: Follow the guidelines above.  Use config files instead of hard‑coding.  Write clear, well‑commented code.  If you need to make assumptions, document them and expose them as parameters.
3. **Testing**: Run unit tests (`pytest`) and script tests (`run_train.py --debug`).  Ensure cross‑platform compatibility by avoiding OS‑specific features.
4. **Documentation (Mandatory)**: Keep documentation fully synced with current behavior. Update docstrings and the relevant `.md` files (usage, algorithm, scientific context). Add new documentation files when needed and link them from the top-level `README.md`. If the change affects the mission or roadmap, coordinate with the doc owners.
5. **Pull Request**: Use conventional commit messages (`feat:`, `fix:`, `docs:`).  Include a description of the change, test results, and any relevant screenshots or sample outputs.  Tag reviewers knowledgeable about data handling or model architecture.

## 9. Notebook Standards

* **Interactive Exploration**: All new or updated `.ipynb` files must be properly interactive and facilitate exploration in the best possible way (use `ipywidgets` where available, and include a clear manual-parameter fallback).
* **Pipeline Choice**: Notebooks that compare algorithms or mixing strategies must expose pipeline selection and key parameters in cells or widgets (no hard-coded-only paths).
* **Documentation**: Every notebook must have exhaustive companion documentation under `docs/` covering purpose, inputs/outputs, dependencies, and usage steps. The top-level `README.md` must link to both the notebook and its documentation.

## 10. Acceptance Checklist (Self‑Verify)

Before marking a task complete or opening a pull request, verify:

- [ ] Code runs in debug and regular modes without raising exceptions.
- [ ] All images remain 16‑bit through the processing pipeline; any conversions are explicit and documented.
- [ ] No absolute paths or secret values appear in the code.  All parameters are configurable.
- [ ] `pytest` passes on Windows and Linux (where possible).  If GPU‑specific code is added, provide a CPU fallback.
- [ ] Logging statements provide sufficient context and do not leak sensitive data.
- [ ] Documentation is fully synced with behavior changes (usage, algorithmic details, scientific assumptions) and is linked appropriately from `README.md`.

By following this AGENTS.md, we ensure that the project remains maintainable, reproducible and aligned with its scientific objectives.
