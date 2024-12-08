import argparse
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
            inputtnet = None
            all_data.append(data)
            all_target.append(target)
            all_output.append(output)

            # Compute loss and metrics
            loss = loss_fn(output, target, data, exptG, combined, config, model)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target, data, 
            #                                inputtnet, combined,
            #                                config, model) * batch_size

    all_data = torch.cat(all_data, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_output = torch.cat(all_output, dim=0)

    # Compute final loss and metrics
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='Config file path')
    args.add_argument('-r', '--resume', default=None, type=str, help='Path to checkpoint')
    args.add_argument('-d', '--device', default=None, type=str, help='GPU indices to enable')

    config = ConfigParser.from_args(args)
    main(config)
