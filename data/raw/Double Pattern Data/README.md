# Double Pattern Data (Experimental)

This folder contains real experimental Kikuchi patterns acquired from a dual-phase steel (BCC + FCC). These images are used to build and validate this project, generate test data, and benchmark mixing/deconvolution algorithms.

> **Data policy:** The original 16-bit BMP files are stored in this folder. Keep them unchanged so experiments and benchmarks remain reproducible.

## Dataset structure

- `50-50 Double Pattern/` mixed patterns where BCC and FCC contributions are approximately 50% each.
- `90-10 Double Pattern/` mixed patterns where BCC and FCC contributions are approximately 90% and 10% respectively.
- `Good Pattern/` single-phase reference patterns:
  - `Perfect_BCC-*.bmp` BCC-only reference patterns
  - `Perfect_FCC-*.bmp` FCC-only reference patterns

## Naming convention

Mixed pattern filenames encode the approximate phase contributions as `BCC-FCC` percentages based on manual inspection. For example:

- `50-50 Double Pattern/50-50_0.bmp` means ~50% BCC and ~50% FCC.
- `90-10 Double Pattern/90-10_3.bmp` means ~90% BCC and ~10% FCC.

These percentages are qualitative estimates intended for algorithm development and validation, not ground-truth labels.

## Intended use

- Evaluate mixing and deconvolution pipelines against real experimental data.
- Build realistic test cases for scripts, notebooks, and metrics.
- Compare synthetic mixing results against observed dual-phase patterns.

## Notes

- Treat these files as raw experimental data; do not overwrite them.
- Preserve the original bit depth and convert to float only during processing.
