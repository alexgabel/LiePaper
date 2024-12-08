import torch.nn.functional as F
import torch


def nll_loss(output, target):
    return F.nll_loss(output, target)

def mse_loss(output, target):
    return F.mse_loss(output, target)

def loss_all(output, target, data, exptG, combined, config, model):
    # Get number of channels from config
    channels = config['arch']['args']['channels']
    bs = data.size(0)
    # batch_tG = torch.einsum('i, mn -> imn', model.t(inputtnet).squeeze(),
    #                         model.G)
    # exp_tG = torch.matrix_exp(batch_tG)

    loss = mse_loss(output, target) \
            + config['lambda_recon'] * (mse_loss(model.normal(data),data) \
                + mse_loss(model.normal(target),target) ) \
            + config['lambda_z'] * mse_loss(model.encoder(target.view(bs, -1)).view(bs,channels,-1), #.mean(dim=2),
                                                        torch.einsum('ica, iba -> icb', 
                                                                    model.encoder(data.view(bs, -1)).view(bs,channels,-1), #.mean(dim=2),
                                                                    exptG
                                                                    )
                                                        ) \
            + config['lambda_lasso'] * torch.norm(model.a, p=1) \
            + config['lambda_a'] * mse_loss(*model.taylor_loss(combined))

    return loss


def loss_multi(output, target, data, exptG, combined, config, model):
    # Get number of channels from config
    channels = config['arch']['args']['channels']
    bs = data.size(0)
    loss = (
        mse_loss(output, target) +
        config['lambda_recon'] * (
            mse_loss(model.normal(data), data) + 
            mse_loss(model.normal(target), target)
        ) +
        config['lambda_z'] * mse_loss(
            model.encoder(target.view(bs, -1)).view(bs, channels, -1),
            torch.einsum('ica, iba -> icb', 
                         model.encoder(data.view(bs, -1)).view(bs, channels, -1),
                         exptG)
        ) +
        config['lambda_lasso'] * (
            torch.norm(model.a, p=1) + torch.norm(model.a2, p=1)
        ) +
        config['lambda_a'] * mse_loss(*model.taylor_loss(combined))
    )

    return loss