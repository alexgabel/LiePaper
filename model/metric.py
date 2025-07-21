import torch
import torch.nn.functional as F

def get_correct_alpha(tf_range, non_affine, device):
    if non_affine:
        alpha = torch.zeros(12, device=device)
        loc_tx, loc_ty = 0, 6
        loc_rot = (2, 7)
        loc_scale = (1, 8)
    else:
        alpha = torch.zeros(6, device=device)
        loc_tx, loc_ty = 0, 3
        loc_rot = (2, 4)
        loc_scale = (1, 5)

    if tf_range[0] != 0:
        alpha[loc_tx] = 1
    if tf_range[1] != 0:
        alpha[loc_ty] = 1
    if tf_range[2] != 0:
        alpha[loc_rot[0]] = -1
        alpha[loc_rot[1]] = 1
    if tf_range[3] != 1:
        alpha[loc_scale[0]] = 1
        alpha[loc_scale[1]] = 1

    return alpha / torch.norm(alpha, p=2), loc_tx, loc_ty, loc_rot, loc_scale

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def mse_loss(output, target, data, inputtnet, combined, config, model):
    return F.mse_loss(output, target)

def mse(x,y):
    return F.mse_loss(x,y)

def loss_recon(output, target, data, inputtnet, combined, config, model):
    loss = mse(model.normal(data),data) \
                + mse(model.normal(target),target)
    return loss

def loss_z(output, target, data, exptG, combined, config, model):
    # Get number of channels from config
    channels = config['arch']['args']['channels']
    bs = data.size(0)
    # batch_tG = torch.einsum('i, mn -> imn', model.t(inputtnet).squeeze(), model.G)
    # print(batch_tG.shape)
    # exp_tG = torch.matrix_exp(batch_tG)

    loss = mse(model.encoder(target.view(bs, -1)).view(bs,channels,-1), #.mean(dim=2),
                torch.einsum('ica, iba -> icb', 
                            model.encoder(data.view(bs, -1)).view(bs,channels,-1), #.mean(dim=2),
                            exptG
                            )
                )

    return loss

def loss_lasso(output, target, data, inputtnet, combined, config, model):
    loss = torch.norm(model.a, p=1) 
    return loss

def loss_alpha(output, target, data, inputtnet, combined, config, model):
    loss = mse(*model.taylor_loss(combined))
    return loss

def generator_mse(output, target, data, inputtnet, combined, config, model):
    alpha_estimate = model.a / torch.norm(model.a, p=2)
    tf_range = config['data_loader']['args']['tf_range']
    non_affine = config['arch']['args']['non_affine']
    correct_alpha, _, _, _, _ = get_correct_alpha(tf_range, non_affine, alpha_estimate.device)
    loss = mse(alpha_estimate, correct_alpha)
    return loss

def drift_mse(output, target, data, inputtnet, combined, config, model):
    alpha_estimate = model.a / torch.norm(model.a, p=2)
    tf_range = config['data_loader']['args']['tf_range']
    non_affine = config['arch']['args']['non_affine']
    correct_alpha, loc_tx, loc_ty, _, _ = get_correct_alpha(tf_range, non_affine, alpha_estimate.device)
    indices = torch.tensor([loc_tx, loc_ty], device=alpha_estimate.device)
    alpha_drift_estimate = torch.index_select(alpha_estimate, 0, indices)
    correct_alpha_drift = torch.index_select(correct_alpha, 0, indices)
    loss = mse(alpha_drift_estimate, correct_alpha_drift)
    return loss

def diffusion_mse(output, target, data, inputtnet, combined, config, model):
    alpha_estimate = model.a / torch.norm(model.a, p=2)
    tf_range = config['data_loader']['args']['tf_range']
    non_affine = config['arch']['args']['non_affine']
    correct_alpha, _, _, loc_rot, loc_scale = get_correct_alpha(tf_range, non_affine, alpha_estimate.device)
    indices = torch.tensor(loc_rot + loc_scale, device=alpha_estimate.device)
    alpha_diff_estimate = torch.index_select(alpha_estimate, 0, indices)
    correct_alpha_diff = torch.index_select(correct_alpha, 0, indices)
    loss = mse(alpha_diff_estimate, correct_alpha_diff)
    return loss