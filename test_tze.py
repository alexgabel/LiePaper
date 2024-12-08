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

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        tf_range=config['data_loader']['args']['tf_range']
    )

    epsilon = 0

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

    # Store all data and output for estimating the generator after:
    all_data = torch.empty(0, 1, 28, 28).to(device)
    all_target = torch.empty(0, 1, 28, 28).to(device)
    all_output = torch.empty(0, 1, 28, 28).to(device)


    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)

            # Add data to all_data tensor:
            all_data = torch.cat((all_data, data), 0)
            all_target = torch.cat((all_target, target), 0)

            #### PREP DATA FOR TRIDENT MODEL ####
            # Combine the data and target tensors into a single tensor:
            combined = torch.cat((data, target), 1)
            #####################################

            output, _ = model(combined, epsilon=epsilon, zTest=True)

            # Add output to all_output tensor:
            all_output = torch.cat((all_output, output), 0)

            ### PLOT ###
            # Plot first image in batch and its output:
            # import matplotlib.pyplot as plt
            # plt.imshow(data[0].cpu().numpy().reshape(28, 28))
            # plt.show()
            # plt.imshow(output[0].cpu().numpy().reshape(28, 28))
            # plt.show()


            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    #### EPSILON TEST ####
    # If epsilon is non-zero, then we need to test the model with epsilon
    # perturbations using the all_data and all_output tensors:
    if epsilon != 0:
        A = all_data.view(-1, data_size)
        B = all_output.view(-1, data_size)
        Target = all_target.view(-1, data_size)
        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        Target = Target.detach().cpu().numpy()
        reg = 1e5

        # Solve for M:
        M = np.linalg.inv(A.T @ A + reg*np.eye(A.shape[1])) @ (A.T @ B)
        # M = np.linalg.lstsq(A, B, rcond=None)[0]

        # Subtract identity matrix and divide by epsilon to get G_est:
        G_est = (M - np.eye(M.shape[0])).T / epsilon

        # Compare G_est to ground truth G:
        D = model.generate_basis(data_size)
        a = torch.tensor([0.0,0.7,0,0.0,0,0.7])
        a_normed = a*1.0 #/ np.linalg.norm(a)
        G_rot = torch.einsum('i, imn -> mn', a_normed, D)

        # Compute the Frobenius norm of the difference between G_est and G_rot:
        G_diff = G_est - G_rot.detach().cpu().numpy()
        G_diff_norm = np.linalg.norm(G_diff, ord='fro')
        print('Frobenius norm of G_diff: {}'.format(G_diff_norm))

        # Compute the coefficients of the linear combination of the basis matrices
        # that make up G_est:
        # G_est_coeffs = np.linalg.lstsq(D, G_est)
        # G_est_coeffs_normed = G_est_coeffs / np.linalg.norm(G_est_coeffs)
        # print('G_est_coeffs: {}'.format(G_est_coeffs_normed))

        # plot both G_rot and G_est:
        # import matplotlib.pyplot as plt
        plt.imshow(G_rot.detach().cpu().numpy(), cmap="seismic", vmin=-1, vmax=1)
        plt.show()
        plt.imshow(G_est, cmap="seismic", vmin=-1, vmax=1)
        plt.show()

        idd = 1
        # plot the target:
        
        for T in [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]:
            # Exponentiate T*G_rot and T*G_est and matrix-vector multiply by a
            # flattened image of a A and B:
            G_rotX = A[idd]@expm(T*G_rot.detach().cpu().numpy()) #torch.einsum('i, imn -> mn', A[0], torch.exp(T*G_rot))
            G_estX = A[idd]@expm(T*G_est) #torch.einsum('i, imn -> mn', A[0], torch.exp(T*G_est))

            G_rotY = B[idd]@expm(T*G_rot.detach().cpu().numpy()) #torch.einsum('i, imn -> mn', A[0], torch.exp(T*G_rot))
            G_estY = B[idd]@expm(T*G_est)

            # Now plot all 6 of the above in a single 2-by-3 grid:
            fig, axs = plt.subplots(2, 4)
            axs[0, 0].imshow(A[idd].reshape((28,28)))
            axs[0, 0].set_title('A')
            axs[0, 1].imshow(G_rotX.reshape((28,28)))
            axs[0, 1].set_title('G_rotX')
            axs[0, 2].imshow(G_estX.reshape((28,28)))
            axs[0, 2].set_title('G_estX')
            axs[1, 0].imshow(B[idd].reshape((28,28)))
            axs[1, 0].set_title('B')
            axs[1, 1].imshow(G_rotY.reshape((28,28)))
            axs[1, 1].set_title('G_rotY')
            axs[1, 2].imshow(G_estY.reshape((28,28)))
            axs[1, 2].set_title('G_estY')
            # Also plot the exponetial of G_rot and G_est:
            axs[0, 3].imshow(Target[idd].reshape((28,28)))
            axs[0, 3].set_title('Target')
            axs[1, 3].imshow(expm(T*G_est).reshape((784,784)))
            axs[1, 3].set_title('exp(T*G_est)')


            for ax in axs.flat:
                ax.label_outer()

            plt.show()

        


        # plot both G_rot and G_est:
        # plt.imshow(G_rotX.reshape((28,28)))
        # plt.show()
        # plt.imshow(G_estX.reshape((28,28)))
        # plt.show()
        # plt.imshow(G_rotY.reshape((28,28)))
        # plt.show()
        # plt.imshow(G_estY.reshape((28,28)))
        # plt.show()


    ######################

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
