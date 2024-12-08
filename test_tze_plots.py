import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from scipy.linalg import expm
import matplotlib.pyplot as plt
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

    data_size = np.prod(data_loader.dataset.images.data[0].shape)

    # build model architecture
    model = config.init_obj('arch', module_arch, input_size=data_size)
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

    all_data, all_target, all_output = [], [], []

    with torch.no_grad():
        for data, target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)

            combined = torch.cat((data, target), 1)
            output, exptG, t = model(combined, epsilon=epsilon)
            all_data.append(data)
            all_target.append(target)
            all_output.append(output)

            # Compute loss
            loss = loss_fn(output, target, data, exptG, combined, config, model)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

    all_data = torch.cat(all_data, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_output = torch.cat(all_output, dim=0)

    # Save results and plots
    # save_dir = os.path.join(config.save_dir, "test_results")
    # os.makedirs(save_dir, exist_ok=True)

    # Generate and save analysis plots
    generate_analysis_plots(all_data, all_target, all_output, model, epsilon)

    # Compute final loss and log
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


def generate_analysis_plots(data, target, output, model, epsilon):
    """
    Generate and display analysis plots for the t-test and epsilon-test.
    """
    # Set up a single figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Analysis Plots", fontsize=16)

    # Flatten axes for easier access
    axes = axes.ravel()

    # t-Test: Visualize different t-values
    t_values = np.linspace(-2.5, 2.5, num=5)  # Fewer t-values for clarity
    for idx, t in enumerate(t_values):
        G = model.G.detach().cpu().numpy()
        exptG = expm(t * G)

        data_sample = data[0].view(-1).detach().cpu().numpy()
        transformed = np.dot(exptG, data_sample).reshape(28, 28)

        # Display transformed image
        axes[idx].imshow(transformed, cmap='gray')
        axes[idx].set_title(f"t-Test t={t:.2f}")
        axes[idx].axis('off')

    # Epsilon-Test: Visualize generator approximation
    if epsilon > 0:
        A = data.view(-1, 28 * 28).detach().cpu().numpy()
        B = output.view(-1, 28 * 28).detach().cpu().numpy()
        reg = 1e-5
        M = np.linalg.inv(A.T @ A + reg * np.eye(A.shape[1])) @ (A.T @ B)
        G_est = (M - np.eye(M.shape[0])) / epsilon

        G_diff = np.linalg.norm(G_est - model.G.detach().cpu().numpy(), ord='fro')

        axes[5].imshow(G_est, cmap="seismic", vmin=-1, vmax=1)
        axes[5].set_title(f"Epsilon-Test | Frobenius Norm: {G_diff:.2f}")
        axes[5].axis('off')

    # Original vs. Output: Visualize a data sample and corresponding output
    sample_idx = 0
    original = data[sample_idx].view(28, 28).detach().cpu().numpy()
    reconstructed = output[sample_idx].view(28, 28).detach().cpu().numpy()
    target_sample = target[sample_idx].view(28, 28).detach().cpu().numpy()

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
    for i in range(len(t_values) + 2, len(axes)):
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
