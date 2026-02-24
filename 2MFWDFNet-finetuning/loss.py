# Loss functions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
import numpy as np
from pytorch_msssim import ssim

# Per-Pixel loss
def Norm_loss(G_img, gt_img, args):
    if args.loss_mode == 'L1':
        metric = torch.abs
    elif args.loss_mode == 'L2':
        metric = torch.square
    else:
        raise ValueError("Mode needs to be L1 or L2")

    loss = torch.mean(metric(G_img - gt_img))

    return loss

# Perceptual loss (VGG19)
def P_loss(G_img, gt_img, vgg_model, args):
    if args.loss_mode == 'L1':
        metric = torch.abs
    elif args.loss_mode == 'L2':
        metric = torch.square
    else:
        assert False, "Mode needs to be L1 or L2"
    
    if not args.color_flag:
        G_img = G_img.repeat(1, 3, 1, 1)  # Repeat along the channel dimension
        gt_img = gt_img.repeat(1, 3, 1, 1)

    preprocessed_G_img = preprocess_input_vgg19(G_img*255.0,'cuda:0')
    preprocessed_gt_img = preprocess_input_vgg19(gt_img*255.0,'cuda:0')

    G_layer_outs = []
    gt_layer_outs = []

    for i in range(len(vgg_model)):

        G_layer_outs.append(vgg_model[i](preprocessed_G_img))
        gt_layer_outs.append(vgg_model[i](preprocessed_gt_img))

    # Compute perceptual loss for each VGG layer output
    loss = sum([torch.mean(metric( (G_layer_out - gt_layer_out)/255. ))
                for G_layer_out, gt_layer_out in zip(G_layer_outs, gt_layer_outs)])
    return loss

# Spatial gradient loss
def Spatial_loss(output_img, GT_img, args):
    if args.loss_mode == 'L1':
        metric = F.l1_loss
    elif args.loss_mode == 'L2':
        metric = F.mse_loss
    else:
        assert False, "Mode needs to be L1 or L2"
    
    def spatial_gradient(x):
        diag_down = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        dv = x[:, :, 1:, :] - x[:, :, :-1, :]
        dh = x[:, :, :, 1:] - x[:, :, :, :-1]
        diag_up = x[:, :, :-1, 1:] - x[:, :, 1:, :-1]
        return [dh, dv, diag_down, diag_up]

    gx = spatial_gradient(output_img)
    gy = spatial_gradient(GT_img)
    
    loss = sum([metric(xx, yy) for xx, yy in zip(gx, gy)])
    return loss

# Loss for the entire end-to-end imaging pipeline
def G_loss(G_img, gt_img, vgg_model, args):
    # Compute metrics
    PSNR = 20 * torch.log10(1.0 / (F.mse_loss(G_img, gt_img))**0.5)
    SSIM = ssim(G_img, gt_img, data_range=1.0, size_average=True)
    metrics = {'PSNR': PSNR, 'SSIM': SSIM}
    
    # Compute losses
    Norm_loss_val = 0.0
    P_loss_val = 0.0
    Spatial_loss_val = 0.0
    if args.Norm_loss_weight != 0.0:
        Norm_loss_val = args.Norm_loss_weight * Norm_loss(G_img, gt_img, args)
    if args.P_loss_weight != 0.0:
        P_loss_val = args.P_loss_weight * P_loss(G_img, gt_img, vgg_model, args)
    if args.Spatial_loss_weight != 0.0:
        Spatial_loss_val = args.Spatial_loss_weight * Spatial_loss(G_img, gt_img, args)
    
    Content_loss_val = Norm_loss_val + P_loss_val + Spatial_loss_val
    loss_components = {'Norm': Norm_loss_val, 'P': P_loss_val, 'Spatial': Spatial_loss_val}
    
    return Content_loss_val, loss_components, metrics

def preprocess_input_vgg19(image, device):
    assert image.max() > 1.0, "图像的值应在 [0, 255] 范围内。"
    mean = torch.tensor([103.939, 116.779, 123.68], device=device).reshape(3, 1, 1).unsqueeze(0)  # BGR channel
    image_bgr = image[:, [2, 1, 0], :, :]
    image_bgr -= mean

    return image_bgr
