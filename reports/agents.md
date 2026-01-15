# reports/AGENTS.md - Reporting Automation Rules

## Mission
Generate **professional, reproducible** slide decks with minimal human effort by treating slides as build artifacts from `report.json` + curated figures.

## Scope
Applies to everything under `reports/` (Quarto decks, build scripts, report schema, CI workflows).

## Non-negotiable rules
- **Never invent metrics, figures, or claims.** Always read `outputs/<run_id>/report.json`.
- **Validate before build.** Run `reports/summarize_results/scripts/validate_report.py`.
- **No screenshots.** Figures must come from pipeline outputs or deterministic scripts in repo.
- **No absolute paths.** `report.json` paths are repo-relative.
- **Deck data source:** `reports/summarize_results/_data/latest.json` only.
- **Required slide structure:**
  1) Title/context (run_id, date, commit, dataset)
  2) What changed since last time (from report notes)
  3) Quantitative results (metrics table)
  4) Loss/metric curves
  5) Qualitative results grid (C, A/B gt, A/B pred, C_hat)
  6) Failure modes (if present)
  7) Next steps (top 5 from `todo_list.md`)

## Build commands
- One-shot build: `scripts/build_summarize_results_report.sh [--run-id <id>] [--strict]`
- Manual: run `build_report.py`, then `quarto render` for PDF + revealjs.

## Acceptance checklist
- [ ] `report.json` validated and used as the only source of metrics.
- [ ] Deck builds with one command and produces `build/deck.pdf` and `build/deck.html`.
- [ ] Title slide includes run_id, timestamp, git commit, dataset tag.
- [ ] Metrics table matches `report.json` exactly.
- [ ] Qual grid shows required panels with labels.
- [ ] No absolute paths and no screenshots.
- [ ] Docs updated if schema or slide structure changes.
