# Cartan: Lie-Symmetric Image Translation

Cartan learns and applies Lie-symmetric transformations to images, pairing an encoder–decoder with a learned generator and a small t-network to estimate transformation parameters. It supports MNIST- or Galaxy10-style grayscale inputs and logs training metrics to TensorBoard.

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
  - Latent dim: `--L 81`
  - Run name: `--n MyRun`

Notebook Demo
- `notebooks/interview_walkthrough.ipynb` demonstrates a mini-batch run: pairing, forward pass, reconstructions, and a quick t-parameter histogram.

Results Artifacts
- Check `saved/` for checkpoints and logs.
- Visuals for the notebook demo are under `notebooks/assets/`.

Acknowledgments
- Parts of the training loop, config parsing, logging utilities, and data-loader scaffolding are adapted from the PyTorch Template by Victor Huang: https://github.com/victoresque/pytorch-template/tree/master
- Those portions retain the original MIT license; see `LICENSE` for details.
