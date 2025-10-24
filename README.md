# Cartan: Type-II Neural Symmetry Detection with Lie Theory

Cartan learns and applies Lie-symmetric transformations to images, pairing an encoder–decoder with a learned generator and a small t-network to estimate transformation parameters. It supports MNIST- or Galaxy10-style grayscale inputs out of the box.

See our paper for more details: https://www.nature.com/articles/s41598-025-17098-8

Quick Start
- Install dependencies: `pip install -r requirements.txt`
- Prepare data under `data/` (e.g., MNIST auto-downloaded; Galaxy10 path in config)
- Train: `python train.py -c config.json`
- Outputs (checkpoints, logs) appear under `saved/`.

Configuration
- Edit `config.json` to adjust architecture, data loader, and training hyperparameters.
- Common CLI overrides:
  - Learning rate: `--lr 1e-3`
  - Batch size: `--bs 512`
  - Latent dim: `--L 25`
  - Run name: `--n run_name`

Acknowledgments
- Parts of the training loop, config parsing, logging utilities, and data-loader scaffolding are adapted from the PyTorch Template by Victor Huang: https://github.com/victoresque/pytorch-template/tree/master
- Those portions retain the original MIT license; see `LICENSE` for details.
