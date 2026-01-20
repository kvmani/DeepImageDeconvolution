# Methods

This document provides a manuscript-style description of the data handling, synthetic mixing, augmentation, training objectives, and evaluation protocol implemented in this repository. It is written to support reproducible scientific reporting.

## Data sources and bit-depth policy

Inputs are grayscale Kikuchi patterns stored as BMP/PNG/JPG/TIF. In practice:

- Most experimental patterns are 16-bit grayscale.
- Some inputs may be 8-bit (legacy exports) or 32-bit containers (typically 8-bit per channel).
- Color inputs (e.g., JPEG) are converted to grayscale using luma weighting before scaling.

All inputs are scaled to a canonical 16-bit grayscale representation for processing, and pipeline outputs are written as 16-bit PNGs. Logging and visualization artifacts remain 8-bit.

The raw demo files under `data/raw/Double Pattern Data/` in this repository are stored as **8-bit grayscale copies** to keep the repo small and documentation-friendly. Always rescale them to 16-bit before processing. If you have original high-bit-depth captures, keep them unchanged outside the repo and use the preparation script to create a canonical training/eval copy under `data/processed/`:

```bash
python3 scripts/prepare_experimental_data.py \
  --input-dir "data/raw/Double Pattern Data" \
  --output-dir "data/processed/Double Pattern Data" \
  --output-format png \
  --output-bit-depth 16
```

The script writes a manifest with checksums and parameters to support reproducibility.
Prepared outputs are always 16-bit PNG files to keep downstream pipelines consistent.

## Preprocessing

Each image is converted to a float representation in the range `[0, 1]`. If the source data are 16-bit, the canonical conversion is:

```
I_float = I_uint16 / 65535
```

If the source data are 8-bit and canonical 16-bit output is requested, we scale by:

```
I_uint16 = round( I_uint8 * 65535 / 255 )
```

### Circular masking

Most EBSD workflows analyze only the detector’s circular active region. We define a mask:

```
M(i, j) = 1  if (i - c_y)^2 + (j - c_x)^2 <= r^2
M(i, j) = 0  otherwise
```

where `(c_y, c_x)` is the image center and `r = min(H, W) / 2`. Masked pixels outside the circle are set to zero. If an input appears already masked (outside pixels near zero), the pipeline records this state.

### Normalization

We support standard min-max and percentile normalization, optionally computed only inside the mask (smart normalization). For min-max:

```
I_norm = (I - I_min) / (I_max - I_min + eps)
```

If `smart` normalization is enabled, `I_min` and `I_max` are computed only on pixels where `M = 1`.

## Synthetic mixing

Given two preprocessed patterns `A` and `B` and a mixing weight `x`:

```
y = 1 - x
C = x * A + y * B
```

Two pipelines are supported:

1. **Normalize-then-mix**:

```
A' = normalize(A)
B' = normalize(B)
C = x * A' + y * B'
```

2. **Mix-then-normalize**:

```
C_raw = x * A + y * B
C = normalize(C_raw)
```

Optional blur/noise can be applied after mixing to approximate detector point spread and measurement noise.

## Augmentation (physics-aware)

Augmentations are disabled by default and can be enabled once baseline behavior is validated. The recommended physics-aware augmentations include:

- **Small rotations and translations** to simulate minor detector alignment shifts.
- **Gain/offset drift** to reflect intensity scale changes across scans.
- **Gaussian blur** to approximate point spread.
- **Gaussian or Poisson noise** to approximate detector noise.

Large flips or 90-degree rotations should be used cautiously and only when physically justified for the EBSD setup.

## Training objective

The dual-output U-Net predicts `A_hat`, `B_hat`, and the mixing weight `x_hat`. The reconstruction is:

```
C_hat = x_hat * A_hat + (1 - x_hat) * B_hat
```

The composite loss is:

```
L = lambda_ab * ( ||A_hat - A||_1 + ||B_hat - B||_1 )
  + lambda_recon * ||C_hat - C||_1
  + lambda_x * ||x_hat - x||_1
```

Hyperparameters `lambda_ab`, `lambda_recon`, and `lambda_x` are set in the training config.

## Evaluation protocol (masked metrics)

Evaluation metrics are computed both globally and within the circular mask. Masked mean squared error is:

```
MSE_mask = sum( M * (P - T)^2 ) / sum(M)
```

Masked PSNR is:

```
PSNR_mask = 20 * log10(R) - 10 * log10(MSE_mask + eps)
```

where `R` is the data range (1.0 for normalized images).

Masked SSIM is computed using mask-weighted local statistics so that pixels outside the circular region do not inflate similarity. This is implemented by normalizing local means and variances with the Gaussian-weighted mask sum.

Report masked metrics alongside global metrics to avoid bias from the zero-valued masked region.

## Limitations

- The synthetic mixing assumes a linear additive model, which is an approximation of EBSD physics.
- Evaluation metrics do not yet include orientation indexing or crystallographic error measures.
- Dataset coverage is limited to the patterns present in `data/raw/Double Pattern Data/` unless augmented with external data.

These limitations are tracked in `todo_list.md` and the long-term roadmap.
