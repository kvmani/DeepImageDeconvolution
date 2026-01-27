# Inference GUI

The inference GUI provides a desktop workflow for running deconvolution on single mixed patterns or
batch folders while staying fully compatible with the existing CLI pipeline. It loads the same YAML
configuration, applies 16-bit preprocessing rules, and writes canonical 16-bit PNG outputs along
with run manifests and optional metrics exports.

## Launching the GUI

Install dependencies (including PySide6):

```bash
pip install -r requirements.txt
```

Run the GUI:

```bash
python3 scripts/run_inference_gui.py
```

## Configuration precedence

The GUI loads a YAML configuration (default: `configs/infer_default.yaml`) and applies UI overrides
on top of it. Precedence is:

1. CLI arguments (when using the CLI scripts)
2. GUI overrides
3. YAML defaults

The GUI does **not** modify the YAML file; it only applies overrides in memory for the current run.

## Inputs

### Mixed pattern input (C)

- **Single image mode** (default): select one mixed pattern image.
- **Batch mode**: select a folder plus a glob pattern (e.g., `*.png`). Toggle recursion to scan
  nested folders.

Supported formats: PNG/TIFF/BMP/JPG. Non-16-bit inputs are scaled to a canonical 16-bit range on
load.

### Optional ground truth (A, B)

Provide GT inputs to compute metrics for `A_hat` vs `A` and `B_hat` vs `B`.

- **Single image mode**: choose explicit A/B files.
- **Batch mode**: choose A/B folders plus a glob pattern. Matching uses the sample ID derived from
  file stems (suffixes like `_A`, `_B`, and `_C` are stripped automatically).

## Outputs

The GUI writes outputs to the configured output directory:

```
<out_dir>/
  A/           # predicted A_hat (16-bit PNG)
  B/           # predicted B_hat (16-bit PNG)
  C_hat/       # reconstructed C (optional, 16-bit PNG)
  run_manifest.json
  metrics.csv  # only when GT is provided
  metrics.json # only when GT is provided
```

### run_manifest.json

The manifest captures:

- effective configuration (YAML + GUI overrides)
- checkpoint path and device
- timing, processed/failed counts, and output counts
- git commit hash (when available)

### Metrics exports

If GT is supplied, the GUI computes per-image metrics and writes:

- `metrics.csv`: per-sample metrics (`a_l1`, `a_l2`, `a_psnr`, `a_ssim`, `b_l1`, `b_l2`, `b_psnr`,
  `b_ssim`)
- `metrics.json`: summary averages and the per-sample table

## Visualization

The viewer uses a linked 2×2 grid:

- **Top-left**: input C
- **Top-right**: predicted A_hat
- **Bottom-left**: predicted B_hat
- **Bottom-right**: diagnostics (metrics + diff map when GT exists)

Use the percentile sliders to adjust contrast for quick inspection. Pixel readouts show the current
cursor position and value. All preview images are rendered as 8-bit for speed while the underlying
data retains 16-bit fidelity.

## Logging

The GUI uses the shared project logging utilities and exposes a log panel with:

- level filtering (DEBUG/INFO/WARNING/ERROR)
- copy, save, and clear actions
- full tracebacks for exceptions

## Notes and tips

- Model loading is lazy and cached across runs. Switching checkpoints triggers a reload, while
  repeated runs with the same settings reuse the cached model.
- Use the **Cancel** button to stop long batch runs; partial outputs and manifests are still saved.

For CLI usage, see `docs/training_inference.md`.
