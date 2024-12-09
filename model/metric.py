import torch
import torch.nn.functional as F

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

    if non_affine: # initialize empty tensor of length 12
        correct_alpha = torch.zeros(12) 
        # order: [Lx, xLx, yLx, x2Lx,  xyLx, y2Lx, Ly, xLy, yLy, x2Ly,  xyLy, y2Ly]
        loc_tx = 0
        loc_ty = 6
        loc_rot = (2, 7)
        loc_scale = (1, 8)
    else:
        correct_alpha = torch.zeros(6)
        # order: [Lx, xLx, yLx, Ly, xLy, yLy]
        loc_tx = 0
        loc_ty = 3
        loc_rot = (2, 4)
        loc_scale = (1, 5)

    # For each non-zero value in tf_range, add coefficients to 
    # the correct_alpha tensor
    if tf_range[0] != 0:
        correct_alpha[loc_tx] = 1
    if tf_range[1] != 0:
        correct_alpha[loc_ty] = 1
    if tf_range[2] != 0:
        correct_alpha[loc_rot[0]] = -1
        correct_alpha[loc_rot[1]] = 1
    if tf_range[3] != 1:
        correct_alpha[loc_scale[0]] = 1
        correct_alpha[loc_scale[1]] = 1


    correct_alpha = correct_alpha / torch.norm(correct_alpha, p=2)
    loss = mse(alpha_estimate.cpu(), correct_alpha)
    return loss

def drift_mse(output, target, data, inputtnet, combined, config, model):
    alpha_estimate = model.a / torch.norm(model.a, p=2)

    tf_range = config['data_loader']['args']['tf_range']
    non_affine = config['arch']['args']['non_affine']

    dev = alpha_estimate.device

    if non_affine: # initialize empty tensor of length 12
        correct_alpha = torch.zeros(12, device=dev) 
        # order: [Lx, xLx, yLx, x2Lx,  xyLx, y2Lx, Ly, xLy, yLy, x2Ly,  xyLy, y2Ly]
        loc_tx = 0
        loc_ty = 6
        loc_rot = (2, 7)
        loc_scale = (1, 8)
    else:
        correct_alpha = torch.zeros(6, device=dev)
        # order: [Lx, xLx, yLx, Ly, xLy, yLy]
        loc_tx = 0
        loc_ty = 3
        loc_rot = (2, 4)
        loc_scale = (1, 5)

    # For each non-zero value in tf_range, add coefficients to 
    # the correct_alpha tensor
    if tf_range[0] != 0:
        correct_alpha[loc_tx] = 1
    if tf_range[1] != 0:
        correct_alpha[loc_ty] = 1
    if tf_range[2] != 0:
        correct_alpha[loc_rot[0]] = -1
        correct_alpha[loc_rot[1]] = 1
    if tf_range[3] != 1:
        correct_alpha[loc_scale[0]] = 1
        correct_alpha[loc_scale[1]] = 1


    correct_alpha = correct_alpha / torch.norm(correct_alpha, p=2)

    # Create index tensor on the same device
    indices = torch.tensor([loc_tx, loc_ty], device=dev)

    # Calculate the loss between entries loc_tx, loc_ty of 
    # both alphas only
    alpha_drift_estimate = torch.index_select(alpha_estimate, 0,indices)
    correct_alpha_drift = torch.index_select(correct_alpha, 0,indices)

    loss = mse(alpha_drift_estimate.cpu(), correct_alpha_drift.cpu())
    return loss

def diffusion_mse(output, target, data, inputtnet, combined, config, model):
    alpha_estimate = model.a / torch.norm(model.a, p=2)
    tf_range = config['data_loader']['args']['tf_range']
    non_affine = config['arch']['args']['non_affine']

    dev = alpha_estimate.device

    if non_affine: # initialize empty tensor of length 12
        correct_alpha = torch.zeros(12, device=dev) 
        # order: [Lx, xLx, yLx, x2Lx,  xyLx, y2Lx, Ly, xLy, yLy, x2Ly,  xyLy, y2Ly]
        loc_tx = 0
        loc_ty = 6
        loc_rot = (2, 7)
        loc_scale = (1, 8)
    else:
        correct_alpha = torch.zeros(6,  device=dev)
        # order: [Lx, xLx, yLx, Ly, xLy, yLy]
        loc_tx = 0
        loc_ty = 3
        loc_rot = (2, 4)
        loc_scale = (1, 5)

    # For each non-zero value in tf_range, add coefficients to 
    # the correct_alpha tensor
    if tf_range[0] != 0:
        correct_alpha[loc_tx] = 1
    if tf_range[1] != 0:
        correct_alpha[loc_ty] = 1
    if tf_range[2] != 0:
        correct_alpha[loc_rot[0]] = -1
        correct_alpha[loc_rot[1]] = 1
    if tf_range[3] != 1:
        correct_alpha[loc_scale[0]] = 1
        correct_alpha[loc_scale[1]] = 1


    correct_alpha = correct_alpha / torch.norm(correct_alpha, p=2)
    indices = torch.tensor(loc_rot + loc_scale, device=dev)
    # Calculate the loss between entries loc_tx, loc_ty of 
    # both alphas only
    alpha_diff_estimate = torch.index_select(alpha_estimate, 0,indices)
    correct_alpha_diff = torch.index_select(correct_alpha, 0,indices)
    

    loss = mse(alpha_diff_estimate.cpu(), correct_alpha_diff.cpu())
    return loss