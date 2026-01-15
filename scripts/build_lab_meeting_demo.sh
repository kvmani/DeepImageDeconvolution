#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

mkdir -p "$QUARTO_HOME"
if [[ -d "$TINY_TEX_BIN" ]]; then
  export PATH="$TINY_TEX_BIN:$PATH"
fi

${PYTHON} "$ROOT_DIR/reports/lab_meeting_demo/scripts/run_demo.py" --strict
HOME="$QUARTO_HOME" quarto render "$ROOT_DIR/reports/lab_meeting_demo/deck.qmd" --to pdf

echo "PDF: $ROOT_DIR/reports/lab_meeting_demo/build/deck.pdf"
if [[ "$BUILD_HTML" -eq 1 ]]; then
  HOME="$QUARTO_HOME" quarto render "$ROOT_DIR/reports/lab_meeting_demo/deck.qmd" --to revealjs
  echo "HTML: $ROOT_DIR/reports/lab_meeting_demo/build/deck.html"
fi
