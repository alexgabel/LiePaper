# Cartan: Neural Symmetry Detector with Latent Lie Algebras

Cartan is a framework for Neural Symmetry Detection in PyTorch. It uses an encoder–decoder style architecture to (1) determine a symmetry generator and (2) estimate transformation magnitudes in order to find geometric relationships in data. It supports MNIST- or Galaxy10-style grayscale inputs out of the box.

See our paper for more details: https://www.nature.com/articles/s41598-025-17098-8

<p align="center">
  <picture>
    <source srcset="docs/cartan_arch.webp" type="image/webp">
    <img src="docs/cartan_arch.png" alt="Cartan model architecture" width="720">
  </picture>
</p>

## Model Overview
- `EncoderLieTDecoder` is the primary architecture: an encoder produces a latent pair, a symmetry generator is parameterized using a basis `D`, and a t-network predicts magnitudes of the transformation. The magnitudes are multiplied by the normalilzed generator and exponentiated before decoding back to the image space. The figure above highlights this pipeline.
- Latent representations can be patch-based (default) or vector-based via `EncoderLieMulTVecDecoder`—choose the variant by swapping the `arch` section in your config.

## Quick Start
- Install dependencies: `pip install -r requirements.txt`
- Prepare data under `data/` (MNIST auto-downloads; Galaxy10 paths provided in configs)
- Train: `python train.py -c config.json`
- Outputs (checkpoints, logs, TensorBoard traces) are written to `saved/`.

## Configuration
- Edit `config.json` (or pass CLI overrides) to set architecture, data loader, and regularisation weights.
- Handy CLI overrides:
  - Learning rate: `--lr 1e-3`
  - Batch size: `--bs 512`
  - Latent dimension (patch): `--L 25`
  - Run name: `--n cartan_run`
- Switch to latent-vector mode by pointing `arch.type` to `EncoderLieMulTVecDecoder` and using `config_vec.json` as a template.

## Experiment Automation & Evaluation Scripts
- `train.sh` – quick sweep runner for a single set of hyperparameters.
- `test.py` – minimal baseline metric evaluation on held-out data.
- `test_tze.py`, `test_tze_plots.py` – richer diagnostics that log generator histograms, transformation grids, etc.
- Results are stored under `images/` for quick inspection.

## Additional Resources
- `sct.ipynb` – demonstrates special conformal transforms on MNIST, useful for stress-testing non-affine behaviour.
- `time_complexity.py` – benchmarks different matrix-exponential routines (`torch.matrix_exp`, truncated series, custom approximations).

## Acknowledgments
- Parts of the training loop, config parsing, logging utilities, and data-loader scaffolding are adapted from the PyTorch Template by Victor Huang: https://github.com/victoresque/pytorch-template/tree/master
- Those portions retain the original MIT license; see `LICENSE` for details.
