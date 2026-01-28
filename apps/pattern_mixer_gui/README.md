# Kikuchi Pattern Mixing Playground GUI

This app provides a local, self-contained GUI for exploring mixing strategies and noise effects on EBSD Kikuchi patterns. It loads two patterns (A and B), applies configurable noise/normalization pipelines, and renders the mixed result (C) live.

> **Bit-depth policy:** Inputs are scaled to a canonical 16-bit range on load and processed internally as `float32` in `[0, 1]`. Exported mixed images are saved as 16-bit PNGs. All displays are 8-bit visualization only.

## Installation

From the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m apps.pattern_mixer_gui
```

or

```bash
python apps/pattern_mixer_gui/main.py
```

Optional logging flags:

```bash
python -m apps.pattern_mixer_gui --log-level INFO --log-file mixer.log
```

## Features

- Load two PNG/BMP/JPG/TIF patterns (RGB inputs are converted to luminance).
- Live mixing with weight `X` (and `Y = 1 - X`).
- Multiple mixing strategies and normalization pipelines.
- Per-image noise controls (Gaussian, Poisson-like shot noise, additive offset).
- High-quality image viewers with zoom/pan, fit/1:1, and per-viewer contrast levels.
- Export mixed C to a 16-bit PNG.
- Log panel for runtime diagnostics.

## Mixing strategies

1. **Normalize A/B then mix**: Applies the chosen normalization to A and B before mixing.
2. **Normalize A/B then mix then normalize C**: Normalizes A and B, mixes, then normalizes C.
3. **Mix raw then normalize C**: Mixes raw intensities first, then normalizes the result.
4. **No normalization (demo)**: Raw mix with no normalization (for demonstration only).

### Normalization modes

- **Min/Max (mask-aware)**: Uses min/max within the circular mask (default).
- **Per-image min/max**: Explicit per-image min/max scaling.
- **Z-score remap**: Z-score per image, then remap to `[0, 1]` using a ±3σ window.
- **No normalization**: Skip normalization entirely.

### Notes on normalization

- Min/max and z-score statistics are computed inside the circular mask when masking is enabled (default).
- Z-score remap uses a fixed ±3σ window to map values to `[0, 1]` and clips outliers.
- If A/B sizes differ, B is resized to match A using bilinear interpolation (logged).

## Noise model notes

- **Gaussian**: Adds zero-mean Gaussian noise with configurable sigma (in `[0, 1]` units).
- **Poisson-like**: Uses a simple Poisson sampling on `image * scale` (scale configurable). This is an approximation, not a full detector noise model.
- **Offset**: Adds a constant background term.

## Screenshot instructions

When making UI changes, capture a screenshot after loading two patterns and adjusting mixing/contrast settings. The GUI is image-first; keep controls compact and focus the capture on the three viewers.
