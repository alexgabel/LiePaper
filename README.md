# Cartan: Type-II Neural Symmetry Detection with Lie Theory

Cartan learns and applies Lie-symmetric transformations to images, pairing an encoder–decoder with a learned generator and a small t-network to estimate transformation parameters. It supports MNIST- or Galaxy10-style grayscale inputs out of the box.

See our paper for more details: https://www.nature.com/articles/s41598-025-17098-8

<p align="center">
  <picture>
    <source srcset="docs/cartan_arch.webp" type="image/webp">
    <img src="docs/cartan_arch.png" alt="Cartan model architecture" width="720">
  </picture>
</p>

## Model Overview
- `EncoderLieTDecoder` is the primary architecture: a Siamese encoder produces a latent pair, a generator basis `D` is learnt, and a t-network predicts the Lie parameters that exponentiate the basis before decoding back to the image space. The figure above highlights this pipeline.
- Latent representations can be patch-based (default) or vector-based via `EncoderLieMulTVecDecoder`—choose the variant by swapping the `arch` section in your config.

## Quick Start
- Install dependencies: `pip install -r requirements.txt`
- Prepare data under `data/` (MNIST auto-downloads; Galaxy10 paths provided in configs)
- Train: `python train.py -c config.json`
- Outputs (checkpoints, logs, TensorBoard traces) are written to `saved/`.

## Configuration & Variants
- Edit `config.json` (or pass CLI overrides) to set architecture, data loader, and regularisation weights.
- Handy CLI overrides:
  - Learning rate: `--lr 1e-3`
  - Batch size: `--bs 512`
  - Latent dimension (patch): `--L 25`
  - Run name: `--n cartan_run`
- Switch to latent-vector mode by pointing `arch.type` to `EncoderLieMulTVecDecoder` and using `config_vec.json` as a template.

## Notebooks & Diagnostics
- `notebooks/interview_walkthrough.ipynb` – guided mini-demo with plots.
- `sct.ipynb` – demonstrates special conformal transforms on MNIST, useful for stress-testing non-affine behaviour.
- `time_complexity.py` – benchmarks different matrix-exponential routines (`torch.matrix_exp`, truncated series, custom approximations).

## Experiment Automation
- `train.sh` – quick sweep runner for a single set of hyperparameters.
- `grid.py` / `grid_search.py` and the `run_grid_search*.sh` scripts – compose configuration grids and launch batch experiments.
- Adjust `config_template.json` and CLI overrides to generate new study suites.

## Evaluation Scripts
- `test.py` – baseline metric evaluation on held-out data.
- `test_tze.py`, `test_tze_new.py`, `test_tze_plots.py` – richer diagnostics that log generator histograms, transformation grids, and Wasserstein distances for `t`, `t₂`, and commutator terms.
- Results are stored under `images/` for quick inspection.

## Acknowledgments
- Parts of the training loop, config parsing, logging utilities, and data-loader scaffolding are adapted from the PyTorch Template by Victor Huang: https://github.com/victoresque/pytorch-template/tree/master
- Those portions retain the original MIT license; see `LICENSE` for details.
