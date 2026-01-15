# reports/AGENTS.md - Reporting Automation Rules

## Mission
Generate professional, reproducible slide decks with minimal human effort by treating slides as build artifacts from report.json plus curated figures.

## Scope
Applies to everything under `reports/` (Quarto decks, build scripts, report schema, CI workflows).

## Shared non-negotiables
- Never invent metrics, figures, or claims. Always read the relevant `outputs/<run_id>/report.json`.
- Validate before build using the report-specific validator.
- No screenshots. Figures must come from pipeline outputs or deterministic scripts in repo.
- No absolute paths in `report.json`.
- Decks must be reproducible via a single build command.

## summarize_results workflow rules
- Validate with `reports/summarize_results/scripts/validate_report.py`.
- Deck data source: `reports/summarize_results/_data/latest.json` only.
- Required slide structure:
  1) Title/context (run_id, date, commit, dataset)
  2) What changed since last time (from report notes)
  3) Quantitative results (metrics table)
  4) Loss/metric curves
  5) Qualitative results grid (C, A/B gt, A/B pred, C_hat)
  6) Failure modes (if present)
  7) Next steps (top 5 from `todo_list.md`)
- Build command: `scripts/build_summarize_results_report.sh [--run-id <id>] [--strict] [--html]`
- Acceptance checklist:
  - report.json validated and used as the only source of metrics.
  - Deck builds with one command and produces `build/deck.pdf` (HTML only when `--html` is used).
  - Title slide includes run_id, timestamp, git commit, dataset tag.
  - Metrics table matches report.json exactly.
  - Qual grid shows required panels with labels.
  - No absolute paths and no screenshots.
  - Docs updated if schema or slide structure changes.

## lab_meeting_demo workflow rules
- Validate with `reports/lab_meeting_demo/scripts/validate_report.py`.
- Deck data source: `reports/lab_meeting_demo/_data/latest.json` or newest `outputs/lab_meeting_demo_*`.
- Required slide structure:
  1) Title slide (project, run_id, date, commit)
  2) Problem definition with equations
  3) Novelty and challenge
  4) Mixing strategies flow diagrams (normalize->mix and mix->normalize)
  5) Weight sweep demo (10/90, 25/75, 50/50)
  6) Strategy comparison (normalize->mix vs mix->normalize)
  7) Dual U-Net approach diagram
  8) Closing/next steps (top 5 from `todo_list.md`)
- Build command: `scripts/build_lab_meeting_demo.sh [--html]`
- Acceptance checklist:
  - report.json validated and used as the only source of figures/paths.
  - Figures are generated from outputs/ by scripts (no manual edits).
  - Deck builds with one command and produces `build/deck.pdf` (HTML only when `--html` is used).
  - Title slide includes run_id, timestamp, git commit.
  - Weight sweep and strategy comparison figures are readable on a projector.
  - No absolute paths and no screenshots.
  - Docs updated if schema or slide structure changes.
