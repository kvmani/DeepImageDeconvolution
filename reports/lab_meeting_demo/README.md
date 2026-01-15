# Lab Meeting Demo Deck

This folder contains the reproducible lab meeting deck and helper scripts for the data synthesis / pattern mixing demo.

## Quickstart

Run the full demo and build the deck in one command:

```bash
bash scripts/build_lab_meeting_demo.sh
```

This will:
- generate `outputs/lab_meeting_demo_<tag>/report.json`
- create slide-ready figures under `outputs/lab_meeting_demo_<tag>/figures/`
- render `reports/lab_meeting_demo/build/deck.pdf`

Optional HTML (revealjs) output:

```bash
bash scripts/build_lab_meeting_demo.sh --html
```

## Run the demo only

```bash
python3 reports/lab_meeting_demo/scripts/run_demo.py --strict
```

Optional overrides:

```bash
python3 reports/lab_meeting_demo/scripts/run_demo.py \
  --A data/raw/Double\\ Pattern\\ Data/Good\\ Pattern/Perfect_BCC-1.bmp \
  --B data/raw/Double\\ Pattern\\ Data/Good\\ Pattern/Perfect_FCC-1.bmp \
  --strategy weight_sweep=mxnorm \
  --weights 0.10,0.25,0.50
```

The run writes:
- `outputs/lab_meeting_demo_<tag>/demo_artifacts/` (A/B inputs and mixed C images)
- `outputs/lab_meeting_demo_<tag>/figures/` (weight sweep and strategy comparison grids)
- `outputs/lab_meeting_demo_<tag>/report.json`
- `outputs/lab_meeting_demo_<tag>/demo_metadata.json`
- `outputs/lab_meeting_demo_<tag>/manifest.json`

`reports/lab_meeting_demo/_data/latest.json` is updated automatically to point the deck at the newest run.

## Folder layout

```
reports/lab_meeting_demo/
  deck.qmd
  _quarto.yml
  README.md
  figures/        # auto-populated by scripts (do not edit manually)
  _data/          # auto-populated latest.json (do not edit manually)
  build/          # deck.pdf + deck.html
  scripts/
    run_demo.py
    make_figures.py
    write_report.py
    validate_report.py
```

## Report contract (report.json)

Required keys:
- `run_id`, `timestamp`, `git_commit`, `stage`
- `inputs.A_path`, `inputs.B_path`
- `strategies.mix_then_normalize.weights`
- `strategies.normalize_then_mix.example_C_path`
- `figures.weight_sweep_grid`, `figures.strategy_compare_grid`
- `notes` (at least two bullets)

Run validation:

```bash
python3 reports/lab_meeting_demo/scripts/validate_report.py --run-id <run_id>
```

## Quarto installation

Quarto is required to build the PDF and HTML decks. Install from:

- https://quarto.org/docs/get-started/

Ensure `quarto` is on your PATH before running the build script.
PDF output also requires a TeX distribution (TinyTeX is recommended):

```bash
quarto install tinytex
```

## Notes

- Generated artifacts belong under `outputs/` and `reports/lab_meeting_demo/build/`.
- `reports/lab_meeting_demo/figures/` and `reports/lab_meeting_demo/_data/latest.json` are build artifacts and are ignored by git.
