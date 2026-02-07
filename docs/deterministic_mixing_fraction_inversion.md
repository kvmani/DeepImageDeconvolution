# Deterministic Mixing-Fraction Inversion (Non-ML Baseline) — Vision Document

This document defines a **deterministic, non-ML baseline** that complements (not replaces) the ML-first mission in `docs/mission_statement.md`.

The core idea: if we can supply **candidate pure patterns** \(A\) and \(B\) that plausibly explain a mixed Kikuchi pattern \(C\), then we can estimate the **mixing fraction** \(x\) in a physically motivated mixture model and quantify uncertainty. This provides:

- a physics-consistent baseline for validating ML results,
- a quantitative tool to map mixing fractions near grain boundaries,
- a harness to compare **metric robustness** (EBSD-style correlation vs pixel losses) under realistic nuisances.

---

## 1) Scope and primary grain-boundary use case

### What this baseline *does*

1. Given a mixed pattern \(C\) and a **candidate pair** \((A, B)\), estimate \(x \in [0, 1]\) in:
   \[
   \hat{C}(x) = xA + (1-x)B
   \]
2. Rank/compare similarity metrics and preprocessing choices using **synthetic ground-truth** where \(x_{\text{true}}\) is known.
3. For real EBSD scans near grain boundaries, output:
   - best-fit \(x^\*\) (and optional confidence),
   - the best candidate pair \((A^\*, B^\*)\) if multiple candidates are supplied,
   - residual diagnostics indicating where the linear mixture model breaks down.

### What this baseline *does not* do (by design)

- It does **not** infer unknown constituents from scratch; it requires either:
  - known/selected \(A\) and \(B\), or
  - a **candidate bank** of plausible \(A\)/\(B\) patterns to search over.
- It does **not** attempt full EBSD indexing/orientation solving (that remains a separate workflow).

This constraint is intentional: it keeps the baseline deterministic, interpretable, and aligned with EBSD template matching practice.

---

## 2) Data policy (16-bit and masking are non-negotiable)

This baseline inherits the repository’s data discipline (see root `agents.md` and `docs/methods.md`):

1. **16-bit discipline**:
   - inputs may be 8-bit or 16-bit, but must be scaled to a canonical 16-bit range for processing;
   - internal arrays are `float32` in \([0, 1]\) (derived from 16-bit);
   - only visualization artifacts may be 8-bit.
2. **Circular masking**:
   - assume the meaningful signal is in the central circular detector region;
   - assume the detector circle is **centered in the image** for this project;
   - detect if inputs appear already masked; otherwise apply the maximum inscribed circular mask;
   - all metrics and normalization statistics are computed **inside the mask** unless explicitly disabled.

These are required for fair metric comparisons and for stable inversion behavior.

---

## 3) Experimental dataset convention (repo-specific)

To seed controlled experiments with a reproducible convention:

- Use `data/raw/Double Pattern Data/50-50 Double Pattern/` as the initial **real mixed-pattern** \(C\) target set.
- Use `data/raw/Double Pattern Data/Good Pattern/` as the **candidate pure-pattern pool**.
- Any file whose name contains `bcc` (case-insensitive) is treated as **A-type** (phase 1).
- Any file whose name contains `fcc` (case-insensitive) is treated as **B-type** (phase 2).

This is a practical convention for preparing synthetic ground truth and initial baselines; it should remain configurable (e.g., via filename regexes) once implemented.

---

## 4) Formal problem statement (mixing-fraction inversion)

### 4.1 Base forward model (convex linear mixture)

Given images \(A, B, C \in \mathbb{R}^{H \times W}\),
\[
\hat{C}(x) = xA + (1-x)B,\quad x \in [0, 1].
\]

Inversion goal:
\[
x^\* = \arg\min_{x \in [0,1]} L(\hat{C}(x), C)
\quad \text{or} \quad
x^\* = \arg\max_{x \in [0,1]} S(\hat{C}(x), C).
\]

### 4.2 Robustness extension: affine intensity nuisance (recommended)

Real EBSD patterns frequently differ by detector gain/offset and background. To avoid “metric fights” caused by unmodeled intensity drift, we explicitly acknowledge an affine intensity nuisance:

\[
\hat{C}(x; g, o) = g\,(xA + (1-x)B) + o
\]

where \(g\) is gain and \(o\) is offset (scalars), applied within the masked region.

Two acceptable ways to handle this (both should be supported eventually):

1. **Preprocess to remove it** (preferred default): background correction + masked standardization (mean/variance) before scoring.
2. **Include it in the fit** for L2-like losses: for each \(x\), solve \((g,o)\) by least squares on masked pixels.

Rationale: this makes the baseline meaningful on real data without forcing all robustness into one metric.

---

## 5) Preprocessing (part of the “model”)

Preprocessing is not a convenience step; it changes the inversion landscape. Therefore:

- preprocessing must be **identical** for \(A\), \(B\), and \(C\);
- every stage is config-controlled and written into the run manifest;
- all statistics are computed inside the circular mask by default.

### Recommended default recipe (starting point)

1. Circular mask (if not already masked).
2. Background correction:
   - **default: subtractive** background correction using a low-pass filtered estimate (e.g., large Gaussian blur).
   - **optional: divisive** background correction (config-controlled) for cases where multiplicative illumination dominates.
3. Masked standardization:
   - mean-center and variance-normalize **inside the mask**.
4. Optional band emphasis (for robustness studies):
   - DoG / high-pass to emphasize Kikuchi band structure.

Important: high-pass / DoG may create negative values; metrics must support signed data (NCC does; SSIM may require careful configuration).

---

## 6) Metrics to implement (minimum set + EBSD-prioritized)

All metrics must be computed on the **same preprocessed arrays** to keep comparisons fair.

### Sanity metrics (pixelwise)

- L1: mean absolute error (masked)
- L2: mean squared error (masked)

These are not expected to be most robust, but they establish baselines and help debug the pipeline.

### Structure-aware metric

- SSIM (masked or computed on masked-cropped region; exact approach must be documented once chosen)

### Correlation family (primary focus)

- **NCC / Pearson correlation** over masked pixels:
  - robust to affine intensity changes (gain/offset), especially when combined with masked standardization;
  - matches EBSD “pattern matching” practice more closely than L1/L2.
- **EBSD pattern correlation (PC)** via `kikuchipy` (optional dependency):
  - treat as first-class if available, as it encodes domain conventions.

### Optional but strongly recommended: small-shift tolerance

Small translations between experimental patterns and candidates can occur (detector geometry, remapping, cropping). Two supported strategies:

1. **Two-stage alignment** (recommended for stability):
   - coarse search for \(x\) without shift;
   - estimate a single best shift at \(x^\*\) (phase correlation);
   - re-score/refine \(x\) with the shift fixed.
2. **Shift-max scoring** (more expensive, non-smooth):
   - for each \(x\), maximize metric over a small shift window.

Rationale: strategy (1) keeps the objective smoother and cheaper while capturing the dominant nuisance.

### Optional but strongly recommended: small-rotation tolerance

Small in-plane rotations can also occur (geometry differences, remapping, scan-to-template mismatch). The baseline should optionally support a bounded rotation search:

- rotation range: up to **±5°** (hard maximum)
- default range: **±2°**
- enable/disable via config

Recommended evaluation strategy (to control cost and non-smoothness):

1. Estimate \(x^\*\) without rotation (and optionally without translation).
2. Estimate the best rigid alignment (translation + rotation) at \(x^\*\).
3. Re-score/refine \(x\) with alignment fixed.

This keeps the primary \(x\)-search 1D while still addressing a key real-data nuisance.

Rotation resampling default:

- use **bicubic interpolation** (config-controlled), and always record interpolation + padding policy in the run manifest for reproducibility.

---

## 7) Search strategy for \(x\) (robust default + fast special cases)

Because correlation/SSIM objectives can be non-smooth (especially with shift-max scoring), the default should be:

- **coarse-to-fine grid search** over \(x \in [0, 1]\)
  - e.g., 0.02 step, then refine around the best region (0.002, then 0.0005).

For smooth objectives (e.g., fixed-preprocessing L2 with optional affine nuisance handled analytically), we may also support:

- bounded 1D optimizers (golden section / Brent),
- closed-form solutions for special cases (documented explicitly if introduced).

Regardless of method, every run must record:

- the score curve vs \(x\) per metric,
- the chosen \(x^\*\),
- optional uncertainty indicators (e.g., peak sharpness, top-2 margin).

---

## 8) Candidate selection for real grain-boundary patterns (critical for end goal)

Near a grain boundary, \(A\) and \(B\) are typically not “given”; they must be proposed from context. This baseline therefore includes a **candidate selection stage**:

1. Provide a candidate bank for A-type and B-type patterns (e.g., interior pixels, template simulations, or the BCC/FCC pool).
2. For each candidate pair \((A_i, B_j)\), run the \(x\)-inversion and compute a final score.
3. Choose the best \((A^\*, B^\*, x^\*)\) under each metric, and report:
   - best pair id(s),
   - best score(s),
   - runner-up gap (a proxy for ambiguity).

Rationale: without this, the method cannot be applied robustly to real boundary regions where “pure” patterns vary across the scan.

---

## 9) Synthetic benchmark plan (ground-truth, nuisance sweeps)

### 9.1 Synthetic mixing (controlled truth)

For \(x \in \{0, 0.05, \dots, 1.0\}\):

1. Construct a mixture with a **config-selected mixing pipeline** consistent with existing repo generation logic:
   - mix-then-normalize
   - normalize-then-mix
   - “repo-default” (exact behavior = shared generator)
2. Optionally add nuisances (config-driven):
   - background gradient
   - gain/offset
   - noise (Gaussian/Poisson)
   - mild blur
   - small translations

All generated artifacts must include a machine-readable manifest that records:

- source \(A\)/\(B\) paths and identifiers,
- \(x_{\text{true}}\),
- preprocessing + nuisance settings,
- random seed and library versions.

### 9.2 Metric robustness ranking

On synthetic data, compute per-metric error statistics:

- MAE, RMSE, bias/variance of \(x^\*\),
- stratification vs nuisance strength (noise, drift, shift),
- failure/ambiguity rates (e.g., flat curves, multiple maxima).

Outcome: a justified recommendation of default preprocessing + metric(s) + search strategy for the deterministic baseline.

---

## 10) Proposed code scaffolding (future implementation, minimal intrusion)

This is the intended module layout once we begin implementation (kept separate from ML pipelines):

- `src/deterministic_mixing_inversion/`
  - `io.py` (load/save, manifests)
  - `mixing.py` (mixture synthesis and \(\hat{C}(x)\))
  - `preprocess.py` (mask/bg/standardize; reuse `src/preprocessing/` where possible)
  - `metrics/` (L1/L2, SSIM, NCC, optional phase correlation, optional `kikuchipy` PC)
  - `search.py` (grid search + refinement)
  - `runner.py` (single experiment entry point)
  - `reporting/` (JSONL/CSV curves, optional HTML report)

Thin scripts in `scripts/` orchestrate runs and must support `--debug`, `--log-level`, `--log-file`, `--quiet`, and run manifests (consistent with the rest of the repo).

## 10.1) “Don’t reinvent EBSD tooling” policy (emphatic)

Before implementing any pattern-processing, correlation, registration, or EBSD-specific metric logic from scratch, we must first consult and reuse existing open-source tooling where practical.

Minimum libraries to consult (and prefer) for readily available solutions:

- `diffpy` (sometimes written “diifpy” in notes; verify the intended package name when wiring dependencies)
- `kikuchipy`
- `pyebsdindex`
- `hyperspy`

Rationale: this baseline’s goal is **robustness and correctness**, not re-deriving standard practice. Reusing vetted implementations reduces risk and aligns metric behavior with the broader EBSD community (and `kikuchipy` commonly leverages `hyperspy`).

---

## 11) Acceptance criteria (“done” for the baseline)

Minimum acceptable milestone:

1. One command can:
   - select (or accept paths for) \(A\), \(B\), \(C\),
   - run preprocessing,
   - estimate \(x^\*\) with at least NCC + L2,
   - write: `results.jsonl`, score-vs-\(x\) curves, and reconstructed \(\hat{C}(x^\*)\).
2. One command can generate a small synthetic benchmark and output:
   - metric ranking table under configured nuisances,
   - summary plots,
   - a short HTML report for inspection (optional but recommended).
3. Debug mode finishes in < 1 minute and runs deterministically.

Stretch goal (grain-boundary mapping):

- produce an \(x(x,y)\) map over a region of an EBSD scan using candidate selection from interior patterns and report ambiguity zones.

---

## 12) Design changes vs the initial plan (and why)

This section records the deliberate changes introduced to the original “mixing-fraction inversion” outline so you can accept/reject them explicitly.

1. **Added an affine intensity nuisance model \((g,o)\)**.
   - *Why:* gain/offset/background are common in EBSD. If unmodeled, they can dominate score curves and make \(x\) estimates misleading.
   - *Impact:* enables either preprocessing-based invariance (default) or explicit nuisance fitting for L2-like losses.
2. **Elevated “candidate selection” to a first-class stage for real boundary use**.
   - *Why:* near boundaries, \(A\) and \(B\) are rarely known exactly; the practical problem is \((A,B,x)\) selection, not only \(x\) inversion.
   - *Impact:* outputs become more actionable on real data and provide ambiguity diagnostics.
3. **Specified a two-stage shift-handling strategy** (estimate shift at \(x^\*\), then refine).
   - *Why:* shift-max scoring for every \(x\) is expensive and can make objectives jagged; a two-stage approach captures the main nuisance with stable computation.
4. **Made 16-bit + circular masking explicit and mandatory** for this baseline.
   - *Why:* metrics are highly sensitive to masked zeros and bit-depth scaling; enforcing these policies avoids misleading comparisons.
5. **Required score curves (not just best \(x\)) and ambiguity proxies**.
   - *Why:* for grain-boundary maps, confidence matters; flat or multi-peaked curves should be detectable and reportable.
6. **Added explicit small-rotation tolerance (config-controlled, bounded)**.
   - *Why:* grain-boundary candidates can be slightly misregistered vs mixed patterns; allowing bounded rotation can prevent systematic bias in \(x\) and avoid false “metric failures.”
7. **Added an explicit “consult EBSD libraries first” policy**.
   - *Why:* correlation and pattern-processing details are subtle; leveraging established implementations increases correctness and keeps effort focused on the scientific question (robust \(x\) recovery near boundaries).

---

## 13) Open questions / constraints needed to finalize implementation details

Please confirm or decide the following before we turn this vision into code:

1. **Translation window**: what maximum pixel shift should we allow in debug vs regular mode (e.g., ±5 px debug, ±15 px regular)?
2. **Rotation step size**: what angular step should we use for the coarse rotation search (e.g., 0.5°), before optional local refinement?
3. **Score aggregation**: when searching candidate pairs \((A_i, B_j)\), should we select winners per-metric independently, or define a single “default winner” policy (e.g., NCC primary, tie-break by residual energy)?
