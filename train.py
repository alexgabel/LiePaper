import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)

    valid_data_loader = data_loader.split_validation()
    # valid_data_loader is a DataLoader object that contains the validation set
    # The next command prints the number of samples in the validation set:

    # Input size is the number of features in the data, product of all but the 0th dimension sizes:
    print(data_loader.dataset.n)
    data_size = int(data_loader.dataset.n)**2

    # build model architecture with given neuron number list, then print to console
    model = config.init_obj('arch', module_arch, input_size=data_size)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    # model = torch.compile(model)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    
    model_params_without_a = filter(lambda p: p.requires_grad and not p is model.a, model.parameters())
    model_params_a = filter(lambda p: p.requires_grad and p is model.a, model.parameters())
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam([
                {'params': model_params_without_a, 'lr': config['optimizer']['args']['lr']},
                {'params': model_params_a, 'lr': config['optimizer']['args']['lr_a']}
            ])
    # optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()
    # # Name of csv file:
    # name_t_csv = config['name'] + '_t_history' + '.csv'
    # # Add path of saved directory to name of csv file:
    # name_t_csv_file = 'saved/' + name_t_csv
    # # Save the values for t stored in trainer.t_history to a csv file in the saved directory:
    # np.savetxt(name_t_csv_file, trainer.t_history, delimiter=',')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Cartan Trainer')
    args.add_argument('-c', '--config', default="config.json", type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--lra', '--learning_rate_a'], type=float, target='optimizer;args;lr_a'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--n', '--name'], type=str, target='name'),
        CustomArgs(['--L', '--latent_dim'], type=int, target='arch;args;latent_dim'),
        CustomArgs(['--c', '--channels'], type=int, target='arch;args;channels'),
        CustomArgs(['--l_r', '--lambda_recon'], type=float, target='lambda_recon'),
        CustomArgs(['--l_z', '--lambda_z'], type=float, target='lambda_z'),
        CustomArgs(['--l_l', '--lambda_lasso'], type=float, target='lambda_lasso'),
        CustomArgs(['--l_a', '--lambda_a'], type=float, target='lambda_a')
        
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
