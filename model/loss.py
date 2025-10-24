import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mse_loss(output, target):
    return F.mse_loss(output, target)


def pixel_mse_loss(output, target):
    return F.mse_loss(output, target)


def loss_all(output, target, data, exptG, combined, config, model):
    # Get number of channels from config
    channels = config['arch']['args']['channels']
    bs = data.size(0)

    loss = (
        pixel_mse_loss(output, target) +  # Compare reconstructed vs target
        config['lambda_recon'] * (
            pixel_mse_loss(model.normal(data), data) +  # Compare normal(data) to data
            pixel_mse_loss(model.normal(target), target)  # Compare normal(target) to target
        ) +
        config['lambda_z'] * mse_loss(
            model.encoder(target.view(bs, -1)).view(bs, channels, -1),
            torch.einsum(
                'ica, iba -> icb',
                model.encoder(data.view(bs, -1)).view(bs, channels, -1),
                exptG
            )
        ) +
        config['lambda_lasso'] * torch.norm(model.a, p=1) +
        config['lambda_a'] * mse_loss(*model.taylor_loss(combined))
    )
    return loss


def loss_multi(output, target, data, exptG, combined, config, model):
    # Get number of channels from config
    channels = config['arch']['args']['channels']
    bs = data.size(0)

    loss = (
        pixel_mse_loss(output, target) +
        config['lambda_recon'] * (
            pixel_mse_loss(model.normal(data), data) + 
            pixel_mse_loss(model.normal(target), target)
        ) +
        config['lambda_z'] * mse_loss(
            model.encoder(target.view(bs, -1)).view(bs, channels, -1),
            torch.einsum(
                'ica, iba -> icb',
                model.encoder(data.view(bs, -1)).view(bs, channels, -1),
                exptG
            )
        ) +
        config['lambda_lasso'] * (
            torch.norm(model.a, p=1) + torch.norm(model.a2, p=1)
        ) +
        config['lambda_a'] * mse_loss(*model.taylor_loss(combined))
    )
    return loss


def loss_vec(output, target, data, exptG, combined, config, model):
    """
    Loss function for EncoderLieMulTVecDecoder.
    
    Args:
        output: Model output (reconstructed data).
        target: Target data.
        data: Input data.
        exptG: Exponential of t * G (transformation matrix).
        combined: Concatenated input and target data for tNet.
        config: Configuration dictionary.
        model: EncoderLieMulTVecDecoder model instance.
    
    Returns:
        Loss value.
    """
    # Batch size
    bs = data.size(0)

    # Get the number of channels
    channels = config['arch']['args']['channels']

    # Calculate the loss
    loss = (
        # Reconstruction loss for output vs target
        pixel_mse_loss(output, target) +

        # Reconstruction loss for normal data and target
        config['lambda_recon'] * (
            pixel_mse_loss(model.normal(data), data) + 
            pixel_mse_loss(model.normal(target), target)
        ) +

        # Latent space transformation loss
        config['lambda_z'] * mse_loss(
            model.encoder(target.view(bs, -1)).view(bs, channels, -1),
            torch.einsum(
                'ica, iba -> icb', 
                model.encoder(data.view(bs, -1)).view(bs, channels, -1),
                exptG
            )
        ) +

        # L1 regularization for the generator matrix G
        config['lambda_lasso'] * torch.norm(model.a, p=1) #+

        # Taylor approximation loss for transformation closure
        #config['lambda_a'] * mse_loss(*model.taylor_loss(combined))
    )

    return loss
