# Mission Statement: Deconvolving Mixed Kikuchi Patterns

## Overview and Vision

Electron‐backscatter diffraction (EBSD) experiments produce high dimensional **Kikuchi patterns** that encode the crystallographic orientation of polycrystalline materials.  In multi‑phase or textured materials two or more crystal orientations often overlap in a single EBSD scan, yielding a **mixed pattern** that is not easily interpreted.  The goal of this project is to build a reproducible pipeline that **generates synthetic mixed patterns** from known pure patterns and trains a machine learning model to **deconvolute** a mixed pattern back into its constituent pure patterns.

This document sets out the scientific and engineering objectives, describes the data generation strategy and network choices, and articulates the principles that will guide development. It emphasises **16‑bit data fidelity**, reproducibility across Windows/Linux and CPU/GPU environments, and modular design. Throughout, we prioritise **clarity over complexity**—explicit reasoning about corner cases and algorithmic steps is preferred to clever but opaque implementations.

Note: this mission statement describes the intended direction. For current implementation status and near-term tasks, see `docs/status.md` and `todo_list.md`.

### Problem Statement

Given two 16‑bit grayscale Kikuchi patterns **A** and **B**, we form a mixed pattern **C** by combining them with weights **x** and **y**:  

\[C = \mathrm{normalize}(x \cdot A + y \cdot B),\quad x + y = 1,\; x,y \in [0,1].\]

Example experimental patterns in this repository live under `data/raw/Double Pattern Data/` (dual-phase steel: BCC + FCC). For instance, `data/raw/Double Pattern Data/Good Pattern/Perfect_BCC-1.bmp` (A) and `data/raw/Double Pattern Data/Good Pattern/Perfect_FCC-1.bmp` (B) are pure references, while `data/raw/Double Pattern Data/50-50 Double Pattern/50-50_0.bmp` is a mixed pattern.

The task is twofold:

1. **Data generation** – starting from a small set of high quality pure patterns, generate realistic mixed patterns \(C\) and paired ground truth \(A,B,x,y\) data at scale.  The generation procedure must preserve the dynamic range of the input (16‑bit) and simulate real mixing physics (e.g., detector point spread, intensity normalisation, noise).
2. **Pattern deconvolution** – design and train a neural network that, given \(C\), simultaneously predicts \(A\), \(B\), and the mixing weights \(x\), \(y\).  The network should learn the inverse of the linear mixing process while respecting physical constraints (non‑negativity, intensity range) and generalise beyond the synthetic training data.

Successfully solving this problem will enable automated phase separation and orientation analysis in EBSD without manual masking or pattern segmentation.  It also provides a blueprint for **source separation** problems where additive mixtures must be decomposed into constituent signals.

### Notation and Glossary

| Symbol / Term | Definition |
| --- | --- |
| \(A\) | Pure Kikuchi pattern from phase/material A. |
| \(B\) | Pure Kikuchi pattern from phase/material B. |
| \(C\) | Mixed Kikuchi pattern formed by combining \(A\) and \(B\). |
| \(x\) | Mixing weight for \(A\), typically tied to phase fraction or relative backscatter contribution. |
| \(y\) | Mixing weight for \(B\), with \(y = 1 - x\). |
| \(\hat{A}\) | Model prediction of \(A\). |
| \(\hat{B}\) | Model prediction of \(B\). |
| \(\hat{C}\) | Reconstructed mixture from predictions, \(\hat{C} = \hat{x}\hat{A} + (1-\hat{x})\hat{B}\). |
| \(\hat{x}\) | Model prediction of \(x\). |
| \(p\) | Pixel index/location within a Kikuchi pattern. |
| \(\epsilon\) | Additive noise term in the mixing model. |
| \(\mathcal{L}\) | Training loss that combines reconstruction and weight supervision terms. |
| \(\lambda_{ab}, \lambda_{recon}, \lambda_x\) | Loss weights for pure pattern reconstruction, mixture reconstruction, and weight supervision. |
| \(\|\cdot\|_1\) | \(\ell_1\) norm used for reconstruction losses. |
| \(\mathrm{normalize}(\cdot)\) | Intensity rescaling operator that maps raw mixtures into a canonical dynamic range. |
| \(\text{Kikuchi pattern}\) | EBSD diffraction image capturing crystallographic orientation-dependent bands. |
| \(\mathrm{PSNR}\) | Peak signal-to-noise ratio used to assess reconstruction fidelity. |
| \(\mathrm{SSIM}\) | Structural similarity index used to compare image structure. |
| \(\text{U-Net}\) | Encoder-decoder CNN with skip connections used for image-to-image reconstruction. |

### Derivation of the Mixing Model

EBSD detectors record the intensity of backscattered electrons that have undergone diffraction within a near-surface interaction volume. When two phases or orientations contribute to the same interaction volume, the detected intensity at each pixel is the sum of contributions from each phase, weighted by their relative backscatter probability and illuminated volume fraction. Let \(A\) and \(B\) represent the pure-pattern intensity fields (after alignment and masking) for two phases. Under a linear detector response, the raw mixed signal at each pixel \(p\) can be written as:

\[
I(p) = x \, A(p) + y \, B(p) + \epsilon(p),
\]

where \(x\) and \(y\) are the fractional contributions and \(\epsilon\) captures noise. Collecting pixels into an image yields the linear mixture \(x \cdot A + y \cdot B\). EBSD acquisition introduces exposure and gain differences across scans, so we apply an intensity normalization operator to place the mixture into a standard dynamic range suitable for learning and comparison. This yields the working model:

\[
C = \mathrm{normalize}(x \cdot A + y \cdot B).
\]

**Assumptions:**

* **Linearity of signal superposition**: electron counts add without interaction terms.
* **Stable detector response**: the detector response is linear over the relevant intensity range.
* **Spatial alignment**: patterns are co-registered so that band features correspond pixel-wise.
* **Consistent masking**: background regions outside the detector circle are excluded.

**Limitations:**

* **Detector saturation or clipping** breaks linearity at high intensities.
* **Cross-scatter and multiple scattering** can introduce non-additive interactions between phases.
* **Phase-dependent backgrounds** (e.g., phosphor response or depth effects) can bias weights.
* **Nonlinear post-processing** (e.g., histogram equalization) can distort mixture linearity if applied prematurely.

## Data Generation Strategy

### Pure Pattern Acquisition

High quality pure Kikuchi patterns are the foundation for training.  These may come from:

* Experimental EBSD data collected on well‑characterised single crystals or polycrystals where patterns can be isolated.
* Simulated patterns generated via dynamical diffraction calculations.  Synthetic patterns help augment limited experimental data and allow systematic variations of orientation and detector geometry.

Raw inputs may arrive as 16-bit grayscale, but 8-bit or 32-bit container formats can also occur (especially BMP exports). The repository keeps raw files unchanged and provides a non-destructive preparation step (`scripts/prepare_experimental_data.py`) to create a canonical 16-bit grayscale copy for training and evaluation. The first step in every data pipeline is to verify bit-depth and convert to a canonical floating-point representation within \([0,1]\).

### Pre‑processing

Before mixing, pure patterns should be aligned, masked, and normalised.  Typical steps include:

1. **Region of interest selection** – crop to remove detector edges or backgrounds.
2. **Circular masking** – apply the maximum inscribed circular mask centered in the image.  Detect whether an input is already masked (outside pixels ≈ 0) and enforce masking where needed.
3. **Intensity normalisation** – rescale intensities to a common range while preserving contrast.  For masked images, compute normalization statistics **inside the circular mask** to avoid zeros outside the active region from skewing the scale (smart normalization).  This is a project-specific option and should remain configurable.  Both global min–max scaling and histogram equalisation variants should be tested.
4. **Noise reduction** – optionally apply denoising filters (e.g., median or Gaussian) to suppress detector noise without destroying band contrast.

#### Normalization Strategy Comparison

| Strategy | Description | Impact on dynamic range and PSNR/SSIM |
| --- | --- | --- |
| Global min–max | Scale intensities using the global minimum and maximum across the full image. | Maximizes global range but can be skewed by masked zeros; may reduce PSNR/SSIM if the background dominates. |
| Percentile scaling | Scale using lower/upper percentiles (e.g., 1st–99th). | Suppresses outliers to preserve band contrast; often improves PSNR/SSIM stability at the cost of clipping extremes. |
| Histogram equalization | Redistribute intensities to flatten the histogram. | Expands local contrast but can distort absolute intensity relationships; may improve perceived contrast while lowering PSNR/SSIM. |
| Smart inside-mask normalization | Compute min/max or percentiles inside the detector mask only. | Preserves useful dynamic range within the signal region; typically improves PSNR/SSIM for masked EBSD patterns by avoiding zero-padding bias. |

### Synthetic Mixing

Training a supervised model requires paired samples \((C,(A,B,x,y))\).  Following the linear mixing approach described in the literature[^1], we will combine pairs of pure patterns using random weights \(x, y\) drawn from a uniform distribution.  Two mixing pipelines will be investigated:

1. **Normalise then mix:** each pure pattern is intensity‑normalised independently; then a weighted sum is computed; finally the result is rescaled to the 16‑bit range.
2. **Mix then normalise:** a weighted sum of raw intensities is formed first; the mixture is then globally normalised.

In addition, geometric transformations (rotation, flipping) will be applied to generate second patterns (e.g., using a vertical flip to create \(B\) from \(A\)).  Noise or point spread convolution can be added to mimic detector blur.  These strategies approximate real physical convolution while maintaining control over the ground truth separation.

## Network Architecture Choices

Several deep learning architectures can be adapted for the deconvolution task:

### U‑Net and Variants

The **U‑Net** architecture features an encoder–decoder structure with skip connections that recover fine details[^2].  For this project we propose a **dual‑output U‑Net** where a single backbone processes the mixed input and splits into two decoder branches that reconstruct \(A\) and \(B\), alongside a weight head that predicts \(\hat{x}\).  Each branch may share initial layers and diverge near the output to allow specialisation.

### CycleGAN / GAN Approaches

Unpaired image translation networks such as **CycleGAN** use adversarial losses and cycle consistency to learn mappings between two domains[^3].  When paired data are available, adversarial losses can still help sharpen reconstructions and reduce blurriness.  We may experiment with a **conditional GAN** that takes \(C\) as input and produces two outputs, alongside discriminator networks that encourage realistic patterns.

### Attention‑Enhanced Models

Modern vision models incorporate attention modules to capture long‑range dependencies.  For example, the **FBPnet‑Sep** architecture separates feature blocks and uses cross‑attention to improve segmentation accuracy[^4].  Similar attention blocks could be inserted into our U‑Net backbone to better model the diffuse Kikuchi bands.  These modules remain optional, and simpler architectures may suffice if they generalise well.

### Design Principles

* **16‑bit aware**: all convolutional layers must accept float tensors in \([0,1]\).  Quantising to 8‑bit should only occur when saving images for visualisation.
* **Modularity**: architectures live under `src/models/` as self‑contained PyTorch modules with clear interfaces; no global variables or implicit assumptions.
* **Cross‑platform**: training/inference scripts should run on CPU or GPU, using PyTorch device management (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`).
* **Config‑driven**: hyperparameters (channel widths, depth, learning rates) are specified in YAML files rather than hard‑coded.

## Training and Evaluation Strategy

### Training Objectives

The baseline dual‑output U‑Net will be trained with a weighted sum of reconstruction losses:

\[\mathcal{L} = \lambda_{ab} \, (\|\hat{A} - A\|_1 + \|\hat{B} - B\|_1) + \lambda_{recon} \, \|\hat{C} - C\|_1 + \lambda_x \, \|\hat{x} - x\|\; ,\]

where \(\hat{C} = \hat{x}\hat{A} + (1-\hat{x})\hat{B}\). The reconstruction term enforces the physical mixing model, while the weight supervision term encourages accurate estimates of \(x\). Additional terms (e.g., perceptual or adversarial losses) can be added as experiments progress. Hyperparameters \(\lambda_i\) are set via configuration.

### Data Handling

Data loaders (`src/datasets/`) read 16‑bit PNG/TIF/BMP images using `Pillow`, convert to float tensors, and apply augmentations on the fly.  Each sample returns a tuple `(C, (A,B,x))`.  Debug mode will restrict the dataset to a small subset or synthetic dummy images to allow fast iteration.

### Training Pipeline

Training scripts (`src/training/train.py`) will:

1. Parse configuration files to set up model, optimizer, scheduler, and dataset paths.
2. Support both **debug** and **regular** modes.  In debug mode, small synthetic datasets and sensible defaults (e.g., 2 epochs, batch size 2) are used; logging is verbose.
3. Log metrics (loss curves, PSNR, SSIM) to console and optionally to file.  Use deterministic seeds for reproducibility.
4. Save checkpoints and best models under `models/`.

### Evaluation

Evaluation scripts (`src/inference/infer.py` and `src/utils/evaluation.py`) will compute image quality metrics such as **peak signal‑to‑noise ratio (PSNR)** and **structural similarity index (SSIM)**.  For EBSD use cases we also plan to compute orientation differences by indexing the recovered patterns and comparing orientation angles.

## Development Roadmap

A high level roadmap for the project is provided in the companion `roadmap.md`.  It breaks down the work into phases—literature survey, data acquisition/pre‑processing, synthetic mixing, model design, training, evaluation, integration, and documentation—and outlines expected deliverables.  This mission statement should be read alongside the roadmap to understand the overall vision.

## References

This document uses numbered footnotes mapped to BibTeX entries in `docs/references/refs.bib`.

[^1]: Placeholder reference for linear mixing of EBSD patterns (BibTeX key: `linear_mixing_ebsd_2020`).
[^2]: Placeholder reference for U-Net architecture (BibTeX key: `ronneberger2015unet`).
[^3]: Placeholder reference for CycleGAN (BibTeX key: `zhu2017cyclegan`).
[^4]: Placeholder reference for FBPnet-Sep or similar attention model (BibTeX key: `fbpnetsep2021`).
