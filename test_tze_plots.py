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


def main(config):
    logger = config.get_logger('test')

    # Explicitly set epsilon here
    epsilon = 0  # You can adjust this value as needed

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=config['data_loader']['args']['batch_size'],
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=config['data_loader']['args']['num_workers'],
        tf_range=config['data_loader']['args']['tf_range']
    )

    # Dynamically determine image dimensions
    img_size = data_loader.dataset.images.data[0].shape[-1]

    # build model architecture
    model = config.init_obj('arch', module_arch, input_size=img_size ** 2)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

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

    # Provide ground-truth t-distribution here
    # Replace the following example with the actual ground-truth t-values
    real_t_distribution = np.random.normal(0, 1, size=all_t.shape[0])

    # Compare distributions
    wasserstein_score = wasserstein_distance(real_t_distribution, all_t)
    logger.info(f"Wasserstein Distance between real and predicted t-distributions: {wasserstein_score:.4f}")

    # Save results and plots
    generate_analysis_plots(all_data, all_target, all_output, model, epsilon, all_t, img_size)

    # Compute final loss and log
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


def generate_analysis_plots(data, target, output, model, epsilon, t_values, img_size):
    """
    Generate and display analysis plots for the t-test and epsilon-test.
    """
    # Set up a single figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Analysis Plots", fontsize=16)

    # Flatten axes for easier access
    axes = axes.ravel()

    # t-Test: Visualize different t-values
    t_sample_values = np.linspace(-2.5, 2.5, num=5)
    for idx, t in enumerate(t_sample_values):
        G = model.G.detach().cpu().numpy()
        exptG = expm(t * G)

        data_sample = data[0].view(-1).detach().cpu().numpy()
        transformed = np.dot(exptG, data_sample).reshape(img_size, img_size)

        # Display transformed image
        axes[idx].imshow(transformed, cmap='gray')
        axes[idx].set_title(f"t-Test t={t:.2f}")
        axes[idx].axis('off')

    # Histogram of t-values
    axes[5].hist(t_values, bins=30, alpha=0.7, label='Predicted t-values')
    axes[5].set_title("Histogram of Predicted t-values")
    axes[5].set_xlabel("t-value")
    axes[5].set_ylabel("Frequency")
    axes[5].legend()

    # Original vs. Output: Visualize a data sample and corresponding output
    sample_idx = 0
    original = data[sample_idx].view(img_size, img_size).detach().cpu().numpy()
    reconstructed = output[sample_idx].view(img_size, img_size).detach().cpu().numpy()
    target_sample = target[sample_idx].view(img_size, img_size).detach().cpu().numpy()

    axes[6].imshow(original, cmap='gray')
    axes[6].set_title("Original Sample")
    axes[6].axis('off')

    axes[7].imshow(target_sample, cmap='gray')
    axes[7].set_title("Target Sample")
    axes[7].axis('off')

    axes[8].imshow(reconstructed, cmap='gray')
    axes[8].set_title("Reconstructed Output")
    axes[8].axis('off')

    # Hide any unused axes
    for i in range(len(t_sample_values) + 2, len(axes)):
        axes[i].axis('off')

    # Display all plots
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='Config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU indices to enable')

    config = ConfigParser.from_args(args)
    main(config)
