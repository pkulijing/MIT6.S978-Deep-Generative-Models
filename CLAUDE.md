# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a learning repository for MIT 6.S978 Deep Generative Models (Fall 2024) course. It contains detailed course notes in markdown format and Jupyter notebook assignments implementing various generative models.

The repository is primarily educational, containing:
- Lecture notes with mathematical derivations and explanations (in Chinese with some English)
- Reading notes summarizing research papers
- Problem sets implemented as Jupyter notebooks
- Supporting assets (images/diagrams)

## Working with Notebooks

### Running the Assignment Notebook

The main assignment is [pset1.ipynb](pset1.ipynb), which implements AutoEncoders (AE), Variational AutoEncoders (VAE), and Point Cloud VAE for 3D torus data.

To run the notebook:
```bash
jupyter lab pset1.ipynb
# or
jupyter notebook pset1.ipynb
```

### Dependencies

The notebook requires:
- PyTorch 2.x with CUDA support (tested with CUDA 12.8)
- torchvision
- matplotlib
- numpy
- tqdm
- mpl_toolkits (for 3D visualization)

The notebook automatically detects and uses CUDA if available. GPU training is used for the VAE models on MNIST dataset (60,000 samples).

### Dataset

The notebook automatically downloads the MNIST dataset to `./data/` directory on first run.

## Code Architecture

### Assignment 1 (pset1.ipynb) Structure

The notebook is organized into sections with problem implementations:

1. **Preliminary Functions (Section 0)**
   - Data loading for MNIST with batch_size=256
   - Generic training loop with progress tracking
   - Evaluation functions for visualizing reconstructions and latent space

2. **AutoEncoder (Section 1)**
   - Class: `AE` - Basic encoder-decoder architecture
   - Loss: MSE reconstruction loss
   - Implements encoder/decoder with configurable hidden dimensions
   - Final hidden dimension is always 2 for 2D latent space visualization

3. **Variational AutoEncoder (Section 2)**
   - Class: `VAE` - Encoder with reparameterization trick
   - Methods:
     - `encode()` - Returns mean and logvar
     - `reparameterize()` - Samples z using reparameterization trick
     - `decode()` - Generates reconstructions from latent code
   - Two loss implementations:
     - `loss_SGVB` - Monte Carlo estimate of ELBO
     - `loss_KL_wo_E` - Analytical KL divergence (preferred)
   - The notebook verifies both losses converge as sample count increases

4. **Torus Point Cloud VAE (Section 3)**
   - Class: `PointVAE` with `PositionalEncoding3D`
   - Custom `PolarVAE` that outputs spherical coordinates (r, theta, phi) and converts to Cartesian
   - Trains on 3D point cloud data sampled from torus surface
   - Tests reconstruction and latent space interpolation

### Key Implementation Details

- Models use `torch.nn.Sequential` for encoder/decoder construction
- Training uses Adam optimizer with typical learning rate 1e-3 to 1e-4
- The generic `loss_func()` combines reconstruction loss with optional regularization term
- VAE latent dimension must be even (split into mean and logvar)
- All models support `.to(device)` for CUDA training

## Documentation Structure

### Lecture Notes

- [lecture1-introduction.md](lecture1-introduction.md) - Overview of generative models, discriminative vs generative
- [lecture2-vae.md](lecture2-vae.md) - Detailed VAE derivation with ELBO, EM algorithm connection
- [lecture3-auto-regressive.md](lecture3-auto-regressive.md) - Autoregressive models

### Reading Notes

Paper summaries with mathematical derivations:
- [reading1-image-priors.md](reading1-image-priors.md) - Image prior research
- [reading1.1-vae-paper.md](reading1.1-vae-paper.md) - Original VAE paper analysis
- [reading2.1-normalizing-flow-glow.md](reading2.1-normalizing-flow-glow.md) - Glow model
- [reading2.2-i-resnet.md](reading2.2-i-resnet.md) - Invertible ResNets
- [reading3.1-auto-regressive-bengio.md](reading3.1-auto-regressive-bengio.md) - Bengio's autoregressive work

## Common Patterns

- Notes are written in Chinese with English technical terms and equations
- Markdown files contain LaTeX math expressions for formulas
- Code comments in notebooks are primarily in Chinese
- Images referenced in markdown are stored in [assets/](assets/) directory
