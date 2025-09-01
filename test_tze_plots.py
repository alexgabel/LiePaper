import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.linalg import expm
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import random


def set_random_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config, output_dir='images'):
    # Set a random seed
    set_random_seed(42)
    logger = config.get_logger('test')
    out_dir = output_dir

    os.makedirs(out_dir, exist_ok=True)

    epsilon = 0  # Adjustable value for epsilon

    # Setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['args']['num_workers'],
        tf_range=config['data_loader']['args']['tf_range']
    )

    img_size = data_loader.dataset.images.data[0].shape[-1]  # Dynamically determine image dimensions

    # Build model architecture
    model = config.init_obj('arch', module_arch, input_size=img_size ** 2)
    logger.info(model)

    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info(f'Loading checkpoint: {config.resume} ...')
    checkpoint = torch.load(config.resume)
    model.load_state_dict(checkpoint['state_dict'])
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns)).to(device)

    all_data, all_target, all_output, all_t = [], [], [], []

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)

            combined = torch.cat((data, target), 1)
            output, exptG, t = model(combined, epsilon=epsilon)

            all_data.append(data)
            all_target.append(target)
            all_output.append(output)
            all_t.append(t)

            # Compute loss
            loss = loss_fn(output, target, data, exptG, combined, config, model)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

    all_data = torch.cat(all_data, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_output = torch.cat(all_output, dim=0)
    all_t = torch.cat(all_t, dim=0).cpu().numpy()

    # Ground-truth t-distribution in radians: Mixture of Gaussians
    # means_deg = [-90, -45, 0, 45, 90]
    # means_rad = [np.deg2rad(mean) for mean in means_deg]
    # std_dev_rad = np.deg2rad(5)
    U = 1.0
    real_t_distribution = np.random.uniform(-U, U, size=all_t.shape[0])  # Uniform distribution for simplicity
    
    # np.concatenate([
    #     np.random.normal(mean, std_dev_rad, size=all_t.shape[0] // len(means_rad))
    #     for mean in means_rad
    # ])

    # Align distributions (mean-shift and scale-normalize)
    aligned_t_values = (all_t - np.mean(all_t)) / np.std(all_t)
    aligned_real_t_distribution = (real_t_distribution - np.mean(real_t_distribution)) / np.std(real_t_distribution)

    # Compare distributions
    wasserstein_score = wasserstein_distance(aligned_real_t_distribution, aligned_t_values)
    logger.info(f"Wasserstein Distance between ground truth and predicted: {wasserstein_score:.5f}")

    # Generate and save analysis plots
    plot_sample_pair_and_histogram(all_data, all_target, all_output, model, all_t, aligned_t_values, out_dir)
    generate_analysis_plots(all_data, all_target, all_output, model, all_t, aligned_t_values, aligned_real_t_distribution, img_size, wasserstein_score, out_dir)
    generate_rows_of_transformations(all_data, all_target, model, all_t, img_size, out_dir)


def plot_sample_pair_and_histogram(data, target, output, model, all_t, aligned_t_values, out_dir):
    """
    Plot a pair of input images (original and target), the output, reconstruction,
    and a histogram.
    """
    sample_idx = 0  # Select the first sample for illustration
    img_size = data[sample_idx].shape[-1]

    # Process the input image through the model for reconstruction
    with torch.no_grad():
        encoded = model.encoder(data[sample_idx].view(-1).unsqueeze(0).to(model.a.device))
        decoded = model.decoder(encoded).view(img_size, img_size).detach().cpu().numpy()

    # Extract original, target, output, and reconstruction
    original = data[sample_idx].view(img_size, img_size).detach().cpu().numpy()
    target_img = target[sample_idx].view(img_size, img_size).detach().cpu().numpy()
    output_img = output[sample_idx].view(img_size, img_size).detach().cpu().numpy()

    # Create figure with gridspec for wider histogram
    fig = plt.figure(figsize=(14, 2.8))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1.5, 0.1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(5)]
    fig.subplots_adjust(wspace=0.4)

    # Plot original
    axes[0].imshow(original, cmap='viridis')
    axes[0].axis('off')
    axes[0].set_title("Original", fontsize=14)

    # Plot target
    axes[1].imshow(target_img, cmap='viridis')
    axes[1].axis('off')
    axes[1].set_title("Target", fontsize=14)

    # Plot output
    axes[2].imshow(output_img, cmap='viridis')
    axes[2].axis('off')
    axes[2].set_title("Output", fontsize=14)

    # Plot reconstruction
    axes[3].imshow(decoded, cmap='viridis')
    axes[3].axis('off')
    axes[3].set_title("Reconstruction", fontsize=14)

    # Plot histogram
    bins = np.linspace(-3, 3, 100)
    # axes[4].hist(aligned_real_t_distribution, bins=bins, alpha=0.6, label="Ground Truth", color="blue", density=True)
    axes[4].hist(all_t, bins=bins, alpha=0.6, label="t", color="green", density=True)
    axes[4].hist(aligned_t_values, bins=bins, alpha=0.6, label="$\\tilde{t}$", color="orange", density=True)
    
    axes[4].set_xlabel("$t$", fontsize=14)
    axes[4].set_ylabel("Density", fontsize=14)
    axes[4].legend(fontsize=12)
    axes[4].spines['top'].set_visible(False)
    axes[4].spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "sample_pair_and_histogram.png"))
    plt.close()


def generate_rows_of_transformations(data, target, model, all_t, img_size, out_dir, num_rows=5):
    """
    Plot multiple rows of transformations for different samples in a single grid,
    showing the effect of continuously varying t without labels or titles.
    """
    t_tilde_values = np.linspace(-3.14, 3.14, 11)  # 10 evenly spaced t values
    t_values = np.std(all_t) * t_tilde_values + np.mean(all_t)  # Normalize t to get t_tilde
    num_cols = len(t_values) + 2
    fig, axes = plt.subplots(num_rows, num_cols + 1, figsize=(1.8 * (num_cols + 1), 2.0 * num_rows))  # One row per sample

    with torch.no_grad():
        for row_idx in range(num_rows):
            sample_idx = row_idx  # Use the row index as the sample index
            
            original = data[sample_idx].view(img_size, img_size).detach().cpu().numpy()
            target_img = target[sample_idx].view(img_size, img_size).detach().cpu().numpy()

            axes[row_idx, 0].imshow(original, cmap='gray')
            axes[row_idx, 0].axis('off')
            if row_idx == 0:
                axes[row_idx, 0].set_title("Original", fontsize=16)

            axes[row_idx, 1].imshow(target_img, cmap='gray')
            axes[row_idx, 1].axis('off')
            if row_idx == 0:
                axes[row_idx, 1].set_title("Target", fontsize=16)

            axes[row_idx, 2].axis('off')  # blank separator column

            images = []

            for t in t_values:
                # Encode the original image
                encoded = model.encoder(data[sample_idx].view(-1).unsqueeze(0).to(model.a.device))
                z = encoded.view(1, model.c, model.latent_dim)

                # Construct exp(tG)
                G = model.G.detach().cpu().numpy()
                exponent = t * G
                exptG = expm(exponent)

                # Apply transformation
                z_transformed = torch.einsum('ica, iba -> icb', z, torch.tensor(exptG).unsqueeze(0).to(z.device))

                # Decode the transformed latent representation
                z_transformed_flat = z_transformed.view(1, -1)
                decoded = model.decoder(z_transformed_flat)

                # Reshape back to image size
                decoded_image = decoded.view(img_size, img_size).detach().cpu().numpy()
                images.append(decoded_image)

            # Stack images horizontally for the current row
            for col_idx, img in enumerate(images):
                axes[row_idx, col_idx + 3].imshow(img, cmap='gray')
                axes[row_idx, col_idx + 3].axis('off')

        for col_idx, t_val in enumerate(t_tilde_values):
            axes[0, col_idx + 3].set_title(f"$\\tilde{{t}}$={t_val:.2f}", fontsize=16)

    # Remove all labels, title, and whitespace
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.05, wspace=0.03)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "rows_of_transformations.png"))
    plt.close()



def generate_analysis_plots(data, target, output, model, all_t, aligned_t_values, aligned_real_t_distribution, img_size, wasserstein_score, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    # First row: Original, Target, Model Output, Distribution Comparison
    sample_idx = 0
    original = data[sample_idx].view(img_size, img_size).detach().cpu().numpy()
    target_sample = target[sample_idx].view(img_size, img_size).detach().cpu().numpy()
    reconstructed = output[sample_idx].view(img_size, img_size).detach().cpu().numpy()

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title("Original", fontsize=14)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(target_sample, cmap='gray')
    axes[0, 1].set_title("Target", fontsize=14)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(reconstructed, cmap='gray')
    axes[0, 2].set_title(f"Output ($\\tilde{{t}}$={aligned_t_values[sample_idx]:.2f}, $t$={all_t[sample_idx]:.2f})", fontsize=14)
    axes[0, 2].axis('off')

    bins = np.linspace(-3, 3, 100)  # Bins for normalized data
    axes[0, 3].hist(aligned_real_t_distribution, bins=bins, alpha=0.5, label="True", color="blue", density=True)
    axes[0, 3].hist(aligned_t_values, bins=bins, alpha=0.5, label="Pred.", color="orange", density=True)
    axes[0, 3].legend(fontsize=12)
    axes[0, 3].set_title(f"Wasserstein Dist: {wasserstein_score:.3f}", fontsize=14)
    axes[0, 3].set_xlabel("$\\tilde{t}$", fontsize=14)
    axes[0, 3].set_ylabel("Density", fontsize=14)
    axes[0, 3].set_box_aspect(1) # Make the histogram approximately square

    # Second row: Transformations for sampled t values using model.encoder and model.decoder
    t_tilde_sample_values = [-2, -1, 0.18, 1]  # Only positive t-values
    t_sample_values = np.std(all_t) *np.array(t_tilde_sample_values) + np.mean(all_t)
    for t_tilde in t_tilde_sample_values:
        axes[0, 3].axvline(x=t_tilde, color='red', linestyle='--', linewidth=0.8)

    with torch.no_grad():
        for i, t in enumerate(t_sample_values):
            # Encode the original image
            encoded = model.encoder(data[sample_idx].view(-1).unsqueeze(0).to(model.a.device))
            t_tf = (t -np.mean(all_t))/np.std(all_t)
            # Reshape for latent processing
            z = encoded.view(1, model.c, model.latent_dim)

            # Construct exp(tG)
            G = model.G.detach().cpu().numpy()
            exponent = t * G
            exptG = expm(exponent)

            # Apply transformation
            z_transformed = torch.einsum('ica, iba -> icb', z, torch.tensor(exptG).unsqueeze(0).to(z.device))

            # Decode the transformed latent representation
            z_transformed_flat = z_transformed.view(1, -1)  # Flatten for decoding
            decoded = model.decoder(z_transformed_flat)

            # Reshape back to image size
            decoded_image = decoded.view(img_size, img_size).detach().cpu().numpy()

            # Plot the transformed image
            axes[1, i].imshow(decoded_image, cmap='gray')
            axes[1, i].set_title(f"$\\tilde{{t}}$={t_tf:.2f}, ${{t}}$={t:.2f}", fontsize=14)
            axes[1, i].axis('off')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "analysis_plots.png"))
    plt.close()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch')
    args.add_argument('-c', '--config', default=None, type=str, help='Config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU indices to enable')
    args.add_argument('--output_dir', default='images', type=str, help='Directory to save images')

    parsed_args = args.parse_args()
    config = ConfigParser.from_args(args)
    main(config, output_dir=parsed_args.output_dir)
