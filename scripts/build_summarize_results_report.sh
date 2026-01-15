#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID=""
STRICT=0
BUILD_HTML=0
PYTHON="python3"
QUARTO_HOME="$ROOT_DIR/.quarto_home"
REAL_HOME="${HOME:-$ROOT_DIR}"
TINY_TEX_BIN="$REAL_HOME/.TinyTeX/bin/x86_64-linux"

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
  export QUARTO_PYTHON="$PYTHON"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --strict)
      STRICT=1
      shift
      ;;
    --html|--revealjs)
      BUILD_HTML=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  latest_report=$(ls -t "$ROOT_DIR"/outputs/*/report.json 2>/dev/null | head -n 1 || true)
  if [[ -z "$latest_report" ]]; then
    echo "No report.json files found under outputs/." >&2
    exit 1
  fi
  RUN_ID=$(basename "$(dirname "$latest_report")")
fi

build_args=(--run-id "$RUN_ID")
if [[ "$STRICT" -eq 1 ]]; then
  build_args+=(--strict)
fi

mkdir -p "$QUARTO_HOME"
if [[ -d "$TINY_TEX_BIN" ]]; then
  export PATH="$TINY_TEX_BIN:$PATH"
fi

${PYTHON} "$ROOT_DIR/reports/summarize_results/scripts/build_report.py" "${build_args[@]}"
HOME="$QUARTO_HOME" quarto render "$ROOT_DIR/reports/summarize_results/deck.qmd" --to pdf

echo "PDF: $ROOT_DIR/reports/summarize_results/build/deck.pdf"
if [[ "$BUILD_HTML" -eq 1 ]]; then
  HOME="$QUARTO_HOME" quarto render "$ROOT_DIR/reports/summarize_results/deck.qmd" --to revealjs
  echo "HTML: $ROOT_DIR/reports/summarize_results/build/deck.html"
fi
