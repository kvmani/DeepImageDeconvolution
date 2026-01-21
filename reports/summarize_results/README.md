# summarize_results (Quarto reporting scaffold)

This folder turns `outputs/<run_id>/report.json` + monitoring figures into a reproducible PDF slide deck (HTML is optional).

## Quickstart

1. Validate and stage figures:

```bash
python3 reports/summarize_results/scripts/build_report.py --run-id <run_id>
```

2. Render slides:

```bash
quarto render reports/summarize_results/deck.qmd --to pdf
# Optional HTML output:
quarto render reports/summarize_results/deck.qmd --to revealjs
```

Or use the one-shot script:

```bash
scripts/build_summarize_results_report.sh --run-id <run_id>
# Optional HTML output:
scripts/build_summarize_results_report.sh --run-id <run_id> --html
```

Outputs:
- `reports/summarize_results/build/deck.pdf`
- `reports/summarize_results/build/deck.html` (only when `--html` is used)

PDF output requires a TeX distribution (TinyTeX recommended):

```bash
quarto install tinytex
```

## report.json contract

All paths in `report.json` are **repository-relative** (POSIX paths). The builder resolves paths relative to the repo root first, then the run directory as a fallback.

Required fields:

```json
{
  "run_id": "string",
  "timestamp": "string",
  "git_commit": "string",
  "stage": "train" | "infer",
  "dataset": "string",
  "dataset_path": "string",
  "config": "string",
  "metrics": { "<metric>": 0.0 },
  "figures": {
    "loss_curve": "string",    // train only
    "qual_grid": "string",
    "weights_plot": "string",  // optional
    "metrics_curve": "string", // optional
    "failure_modes": ["string"] // optional
  },
  "status": "running" | "complete" | "interrupted" | "failed",
  "progress": {
    "epoch": 0,
    "epochs_total": 0,
    "global_step": 0
  }
}
```

Optional fields:

- `notes`: list of short strings.
- `artifacts`: key/value paths (e.g., `best_ckpt`, `last_ckpt`, `history`, `metrics_csv`, `image_log`, `image_log_html`).
- `tracking_sample`: object with `sample_id`, `epoch`, and per-image paths for A/B prediction tracking (when image logging is enabled).
- `comparison_table`: relative path to a CSV table of recent runs (used by the deck).

## Build artifacts

`build_report.py` creates:
- `reports/summarize_results/_data/latest.json` (data source for the deck)
- `reports/summarize_results/_data/run_comparison.csv` (recent run comparison table)
- `reports/summarize_results/figures/*.png` (local copies of report figures)
- `reports/summarize_results/build/manifest.txt` (copy log)

## Validation

Strict validation (schema + figure paths):

```bash
python3 reports/summarize_results/scripts/validate_report.py --run-id <run_id>
```

Non-strict build creates placeholder images when figures are missing:

```bash
python3 reports/summarize_results/scripts/build_report.py --run-id <run_id>
```

Strict mode fails if any required figure is missing:

```bash
python3 reports/summarize_results/scripts/build_report.py --run-id <run_id> --strict
```
