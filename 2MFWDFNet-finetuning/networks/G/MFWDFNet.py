# Neural feature propagator network
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
from .network_arch import *

def resize_with_crop_or_pad(input, target_height, target_width):
    _, _, height, width = input.shape
    
    if height > target_height:
        crop_top = (height - target_height) // 2
        crop_bottom = crop_top + target_height
    else:
        crop_top = 0
        crop_bottom = height
    
    if width > target_width:
        crop_left = (width - target_width) // 2
        crop_right = crop_left + target_width
    else:
        crop_left = 0
        crop_right = width

    input_cropped = input[:, :, crop_top:crop_bottom, crop_left:crop_right]

    pad_height = max(0, target_height - (crop_bottom - crop_top))
    pad_width = max(0, target_width - (crop_right - crop_left))
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    input_padded = F.pad(input_cropped, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return input_padded

def fft(img):
    Fimg = torch.fft.fft2(img)
    return Fimg

def ifft(Fimg):
    img = torch.fft.ifft2(Fimg)
    img = torch.abs(img)
    img = img.to(torch.float32)
    return img

def psf2otf(psf, h, w):
    psf = resize_with_crop_or_pad(psf,h,w)
    psf = psf.to(torch.complex64)
    psf = torch.fft.fftshift(psf, dim=(2, 3))
    otf = torch.fft.fft2(psf) 
    return otf

def deconvolve_wnr(blur, snr, otf):
    
    blur_debug = blur
    otf_conj = torch.conj(otf)
    otf_abs2 = torch.abs(otf) ** 2
    gamma = 1/snr
    wiener_filter = otf_conj / (otf_abs2.to(torch.complex64) + gamma.to(torch.complex64))

    output_fft = wiener_filter * fft(blur)
    output = torch.abs(torch.fft.ifft2(output_fft)).float()
    return output, blur_debug

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None, apply_instnorm=True):
        super(Conv, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=True)]
        if apply_instnorm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if not activation == None:
            layers.append(activation)
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)

class ConvTransp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None,apply_instnorm=True):
        super(ConvTransp, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=0, bias=True)]
        if not activation == None:
            layers.append(activation)
        self.conv_transp = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_transp(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None):
        super(DoubleConv, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=True)]
        if not activation == None:
            layers.append(activation)
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=True))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, kernel_size, stride, activation)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.double_conv(x)
        return x + self.res_scale * res

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction, activation):
        super(ChannelAttention, self).__init__()
        layers = [nn.AdaptiveAvgPool2d(1)]
        layers.append(nn.Conv2d(num_features, num_features // reduction, kernel_size=1))
        if not activation == None:
            layers.append(activation)
        layers.append(nn.Conv2d(num_features // reduction, num_features, kernel_size=1))
        layers.append(nn.Sigmoid())

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return x * self.module(x)

class RCAB(nn.Module):
    def __init__(self, num_features, reduction, size, activation=None, res_scale=1):
        super(RCAB, self).__init__()
        self.double_conv = DoubleConv(in_channels=num_features, out_channels=num_features, kernel_size=size, stride=1, activation=activation)
        self.channel_attention = ChannelAttention(num_features, reduction, activation)
        self.res_scale = res_scale
    def forward(self, x):
        res_ca = self.channel_attention(self.double_conv(x))
        return x + self.res_scale*res_ca
    

class FeatExtractRCAB(nn.Module):
    def __init__(self):
        super(FeatExtractRCAB, self).__init__()
        self.res_scale = 1
        self.LReLU = nn.LeakyReLU(negative_slope=0.3, inplace=False)

        # Define layers for the downsampled branches
        self.down_l0 = Conv(3, 15, 7, 1, self.LReLU, apply_instnorm=False)
        self.down_l1 = Conv(15, 30, 5, 2, self.LReLU, apply_instnorm=False)
        self.down_l2 = Conv(30, 60, 5, 2, self.LReLU, apply_instnorm=False)

        # Layers for 4x, 2x, and 1x processing
        # 4x
        self.conv_l2_k0 = Conv(60, 60, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l2_k1 = RCAB(60, 15, 3, self.LReLU, res_scale=1)
        self.conv_l2_k2 = RCAB(60, 15, 3, self.LReLU, res_scale=1)
        self.conv_l2_k3 = RCAB(60, 15, 3, self.LReLU, res_scale=1)

        # 2x
        self.conv_l1_k0 = Conv(30, 30, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_l1_k1 = RCAB(30, 15, 3, self.LReLU, res_scale=1)
        self.conv_l1_k2 = RCAB(30, 15, 3, self.LReLU, res_scale=1)
        self.conv_l1_k3 = RCAB(30, 15, 3, self.LReLU, res_scale=1)

        self.up_l2 = ConvTransp(60, 30, 2, 2, self.LReLU, apply_instnorm=False)
        self.conv_l1_k4 = Conv(60, 30, 3, 1, self.LReLU, apply_instnorm=False)    # cat 后c维度翻倍
        self.conv_l1_k5 = Conv(30, 30, 3, 1, self.LReLU, apply_instnorm=False)

        # 1x
        self.conv_l0_k0 = Conv(15, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k1 = RCAB(15, 5, 5, self.LReLU, res_scale=1)
        self.conv_l0_k2 = RCAB(15, 5, 5, self.LReLU, res_scale=1)
        self.conv_l0_k3 = RCAB(15, 5, 5, self.LReLU, res_scale=1)

        self.up_l1 = ConvTransp(30, 15, 2, 2, self.LReLU, apply_instnorm=False)
        self.conv_l0_k4 = Conv(30, 15, 5, 1, self.LReLU, apply_instnorm=False)
        self.conv_l0_k5 = Conv(15, 15, 5, 1, self.LReLU, apply_instnorm=False)

    def forward(self, img):

        # Downsample
        down_l0 = self.down_l0(img)
        down_l1 = self.down_l1(down_l0)
        down_l2 = self.down_l2(down_l1)

        # 4x block
        conv_l2_k0 = self.conv_l2_k0(down_l2)
        conv_l2_k1 = self.conv_l2_k1(conv_l2_k0)
        conv_l2_k2 = self.conv_l2_k2(conv_l2_k1)
        conv_l2_k3 = self.conv_l2_k3(conv_l2_k2)
        img_fp_4x = conv_l2_k3

        # 2x block
        conv_l1_k0 = self.conv_l1_k0(down_l1)
        conv_l1_k1 = self.conv_l1_k1(conv_l1_k0)
        conv_l1_k2 = self.conv_l1_k2(conv_l1_k1)
        conv_l1_k3 = self.conv_l1_k3(conv_l1_k2)

        up_l2 = self.up_l2(conv_l2_k3)
        conv_l1_k4 = self.conv_l1_k4(torch.cat([up_l2, conv_l1_k3], dim=1))
        conv_l1_k5 = self.conv_l1_k5(conv_l1_k4)
        img_fp_2x = conv_l1_k5

        # 1x block
        conv_l0_k0 = self.conv_l0_k0(down_l0)
        conv_l0_k1 = self.conv_l0_k1(conv_l0_k0)
        conv_l0_k2 = self.conv_l0_k2(conv_l0_k1)
        conv_l0_k3 = self.conv_l0_k3(conv_l0_k2)

        up_l1 = self.up_l1(conv_l1_k3)
        conv_l0_k4 = self.conv_l0_k4(torch.cat([up_l1, conv_l0_k3], dim=1))
        conv_l0_k5 = self.conv_l0_k5(conv_l0_k4)
        img_fp_1x = conv_l0_k5

        return img_fp_4x, img_fp_2x, img_fp_1x

class DeconvWNRinFPsinglePSF(nn.Module):
    def __init__(self, params, args):
        super(DeconvWNRinFPsinglePSF, self).__init__()
        self.args = args
        initial_snr = torch.tensor(args.snr_init, dtype=torch.float32)
        if args.snr_opt:
           self.snr = nn.Parameter(initial_snr, requires_grad=True)
           # self.snr = torch.clamp(self.snr, 3.0, 4.0)  # Clamp snr within range
        else:
           self.snr = initial_snr

        real_psf = params['psf']
        psf_1x = real_psf.permute(0, 3, 1, 2)  # (1, c, h, w)

        h = params['load_width']
        w = params['load_width']
        self.otf_1x = psf2otf(psf_1x, h, w)

        psf_2x = F.avg_pool2d(psf_1x, 2)
        psf_2x = psf_2x / torch.sum(psf_2x, dim=[2, 3], keepdim=True)
        self.otf_2x = psf2otf(psf_2x, h // 2, w // 2)

        psf_4x = F.avg_pool2d(psf_1x, 4)
        psf_4x = psf_4x / torch.sum(psf_4x, dim=[2, 3], keepdim=True)
        self.otf_4x = psf2otf(psf_4x, h // 4, w // 4)
        
    def forward(self, img_fp_4x, img_fp_2x, img_fp_1x):
        
        wien_decon_4x, _ = deconvolve_wnr(img_fp_4x, torch.pow(10.0, self.snr), self.otf_4x.repeat(1, 20, 1, 1))
        wien_decon_2x, _ = deconvolve_wnr(img_fp_2x, torch.pow(10.0, self.snr), self.otf_2x.repeat(1, 10, 1, 1))
        wien_decon_1x, _ = deconvolve_wnr(img_fp_1x, torch.pow(10.0, self.snr), self.otf_1x.repeat(1, 5, 1, 1))

        return wien_decon_4x, wien_decon_2x, wien_decon_1x


class DeconvWNRinFPmultiPSF(nn.Module):
    def __init__(self, params, args):
        super(DeconvWNRinFPmultiPSF, self).__init__()
        self.args = args

        real_psf = params['psf']
        h = params['load_width']
        w = params['load_width']

        if args.mode_multi_cpsf == 'psf_no_opt':
           
           real_psf = real_psf.permute(0, 3, 1, 2)  # (1, c, h, w)
           psf_1x = torch.cat([real_psf[i, :, :, :] for i in range(5)], dim=0).unsqueeze(0)

           psf_2x = F.avg_pool2d(psf_1x, 2)
           psf_2x = psf_2x / torch.sum(psf_2x, dim=[2, 3], keepdim=True)

           psf_4x = F.avg_pool2d(psf_1x, 4)
           psf_4x = psf_4x / torch.sum(psf_4x, dim=[2, 3], keepdim=True)
           
           self.otf_1x = psf2otf(psf_1x, h, w)
           self.otf_2x = psf2otf(psf_2x, h // 2, w // 2).repeat(1, 2, 1, 1)
           self.otf_4x = psf2otf(psf_4x, h // 4, w // 4).repeat(1, 4, 1, 1)

           self.snr_1x = nn.Parameter(torch.tensor(args.snr_init*np.ones([1, 15, 1, 1]), dtype=torch.float32), requires_grad=True)
           self.snr_2x = nn.Parameter(torch.tensor(args.snr_init*np.ones([1, 30, 1, 1]), dtype=torch.float32), requires_grad=True)
           self.snr_4x = nn.Parameter(torch.tensor(args.snr_init*np.ones([1, 60, 1, 1]), dtype=torch.float32), requires_grad=True)

           # self.snr = torch.clamp(self.snr, 3.0, 4.0)  # Clamp snr within range

        elif args.mode_multi_cpsf == 'psf_spatial_opt':
           
           real_psf = real_psf.permute(0, 3, 1, 2)
           initial_psf_1x = torch.cat([real_psf[i, :, :, :] for i in range(5)], dim=0).unsqueeze(0)

           initial_psf_2x = F.avg_pool2d(initial_psf_1x, 2)
           initial_psf_2x = initial_psf_2x / torch.sum(initial_psf_2x, dim=[2, 3], keepdim=True)
           initial_psf_2x = initial_psf_2x.repeat(1, 2, 1, 1)

           initial_psf_4x = F.avg_pool2d(initial_psf_1x, 4)
           initial_psf_4x = initial_psf_4x / torch.sum(initial_psf_4x, dim=[2, 3], keepdim=True)
           initial_psf_4x = initial_psf_4x.repeat(1, 4, 1, 1)

           self.psf_1x = nn.Parameter(initial_psf_1x, requires_grad=True)
           self.psf_2x = nn.Parameter(initial_psf_2x, requires_grad=True)
           self.psf_4x = nn.Parameter(initial_psf_4x, requires_grad=True)
           
           self.otf_1x = psf2otf(self.psf_1x, h, w)
           self.otf_2x = psf2otf(self.psf_2x, h // 2, w // 2)
           self.otf_4x = psf2otf(self.psf_4x, h // 4, w // 4)

           initial_snr_1x = torch.tensor(args.snr_init*np.ones([1,15,1,1]), dtype=torch.float32)
           initial_snr_2x = torch.tensor(args.snr_init*np.ones([1,30,1,1]), dtype=torch.float32)
           initial_snr_4x = torch.tensor(args.snr_init*np.ones([1,60,1,1]), dtype=torch.float32)

           self.snr_1x = nn.Parameter(initial_snr_1x, requires_grad=True)
           self.snr_2x = nn.Parameter(initial_snr_2x, requires_grad=True)
           self.snr_4x = nn.Parameter(initial_snr_4x, requires_grad=True)
           
        elif args.mode_multi_cpsf == 'psf_frequency_opt_with_initial':
           
           real_psf = real_psf.permute(0, 3, 1, 2)
           psf_1x = torch.cat([real_psf[i, :, :, :] for i in range(5)], dim=0).unsqueeze(0)

           psf_2x = F.avg_pool2d(psf_1x, 2)
           psf_2x = psf_2x / torch.sum(psf_2x, dim=[2, 3], keepdim=True)
           psf_2x = psf_2x.repeat(1, 2, 1, 1)

           psf_4x = F.avg_pool2d(psf_1x, 4)
           psf_4x = psf_4x / torch.sum(psf_4x, dim=[2, 3], keepdim=True)
           psf_4x = psf_4x.repeat(1, 4, 1, 1)

           otf_1x = psf2otf(psf_1x, h, w)
           otf_2x = psf2otf(psf_2x, h // 2, w // 2)
           otf_4x = psf2otf(psf_4x, h // 4, w // 4)

           # initial_wiener_1x =  torch.conj(otf_1x)/((torch.abs(otf_1x) ** 2).to(torch.complex64) + (1/torch.pow(10.0, params['snr_1x'])).to(torch.complex64))
           # initial_wiener_2x =  torch.conj(otf_2x)/((torch.abs(otf_2x) ** 2).to(torch.complex64) + (1/torch.pow(10.0, params['snr_2x'])).to(torch.complex64))
           # initial_wiener_4x =  torch.conj(otf_4x)/((torch.abs(otf_4x) ** 2).to(torch.complex64) + (1/torch.pow(10.0, params['snr_4x'])).to(torch.complex64))

           initial_wiener_1x =  torch.conj(otf_1x)/((torch.abs(otf_1x) ** 2).to(torch.complex64) + (1/torch.pow(10.0, torch.tensor(args.snr_init, dtype=torch.float32))).to(torch.complex64))
           initial_wiener_2x =  torch.conj(otf_2x)/((torch.abs(otf_2x) ** 2).to(torch.complex64) + (1/torch.pow(10.0, torch.tensor(args.snr_init, dtype=torch.float32))).to(torch.complex64))
           initial_wiener_4x =  torch.conj(otf_4x)/((torch.abs(otf_4x) ** 2).to(torch.complex64) + (1/torch.pow(10.0, torch.tensor(args.snr_init, dtype=torch.float32))).to(torch.complex64))

           self.wiener_1x_re = nn.Parameter(initial_wiener_1x.real, requires_grad=True)
           self.wiener_1x_im = nn.Parameter(initial_wiener_1x.imag, requires_grad=True)

           self.wiener_2x_re = nn.Parameter(initial_wiener_2x.real, requires_grad=True)
           self.wiener_2x_im = nn.Parameter(initial_wiener_2x.imag, requires_grad=True)

           self.wiener_4x_re = nn.Parameter(initial_wiener_4x.real, requires_grad=True)
           self.wiener_4x_im = nn.Parameter(initial_wiener_4x.imag, requires_grad=True)

           self.normalizer_1x = nn.Parameter(torch.ones(1, 15, 1, 1)* 10)
           self.normalizer_2x = nn.Parameter(torch.ones(1, 30, 1, 1)* 10)
           self.normalizer_4x = nn.Parameter(torch.ones(1, 60, 1, 1)* 10)


        elif args.mode_multi_cpsf == 'psf_frequency_opt_no_initial':

           self.wiener_1x_re = nn.Parameter(torch.rand(1, 15, h, w)* 0.001)
           self.wiener_1x_im = nn.Parameter(torch.rand(1, 15, h, w)* 0.001)

           self.wiener_2x_re = nn.Parameter(torch.rand(1, 30, h // 2, w // 2)* 0.001)
           self.wiener_2x_im = nn.Parameter(torch.rand(1, 30, h // 2, w // 2)* 0.001)

           self.wiener_4x_re = nn.Parameter(torch.rand(1, 60, h // 4, w // 4)* 0.001)
           self.wiener_4x_im = nn.Parameter(torch.rand(1, 60, h // 4, w // 4)* 0.001)

           self.normalizer_1x = nn.Parameter(torch.ones(1, 15, 1, 1)* 10)
           self.normalizer_2x = nn.Parameter(torch.ones(1, 30, 1, 1)* 10)
           self.normalizer_4x = nn.Parameter(torch.ones(1, 60, 1, 1)* 10)

    def forward(self, img_fp_4x, img_fp_2x, img_fp_1x):
        
        # Prepare otf and snr for different scales
        if self.args.mode_multi_cpsf == 'psf_no_opt' or self.args.mode_multi_cpsf == 'psf_spatial_opt':

           wien_decon_1x, _ = deconvolve_wnr(img_fp_1x, torch.pow(10.0, self.snr_1x), self.otf_1x)

           wien_decon_4x, _ = deconvolve_wnr(img_fp_4x, torch.pow(10.0, self.snr_4x), self.otf_4x)

           wien_decon_2x, _ = deconvolve_wnr(img_fp_2x, torch.pow(10.0, self.snr_2x), self.otf_2x)


        elif self.args.mode_multi_cpsf == 'psf_frequency_opt_with_initial' or self.args.mode_multi_cpsf == 'psf_frequency_opt_no_initial':

           wiener_1x = torch.complex(self.wiener_1x_re, self.wiener_1x_im)
           wiener_2x = torch.complex(self.wiener_2x_re, self.wiener_2x_im)
           wiener_4x = torch.complex(self.wiener_4x_re, self.wiener_4x_im)
           
           wien_decon_4x = torch.abs(torch.fft.ifft2(wiener_4x * fft(img_fp_4x))*self.normalizer_4x/10).float()

           wien_decon_2x = torch.abs(torch.fft.ifft2(wiener_2x * fft(img_fp_2x))*self.normalizer_2x/10).float()

           wien_decon_1x = torch.abs(torch.fft.ifft2(wiener_1x * fft(img_fp_1x))*self.normalizer_1x/10).float()

        return wien_decon_4x, wien_decon_2x, wien_decon_1x



class FusionNAFnet(nn.Module):
    def __init__(self, params):
        super(FusionNAFnet, self).__init__()

        self.params = params

        ## encoder
        self.conv_l0_k0 = nn.Conv2d(in_channels=15, out_channels=30, kernel_size=5, padding=2, stride=1, groups=1,bias=True)

        self.conv_l0_k1 = NAFBlock(30)
        self.down_l0 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, padding=2, stride=2, groups=1,bias=True)

        self.conv_l1_k1 = NAFBlock(60)
        self.down_l1 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=3, padding=1, stride=2, groups=1,bias=True)

        self.conv_l2_k1 = NAFBlock(120)
        self.conv_l2_k2 = nn.Conv2d(in_channels=240, out_channels=120, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        self.conv_l2_k3 = NAFBlock(120)

        ## decoder
        self.up_l2 = nn.Sequential(nn.Conv2d(in_channels=120, out_channels=240, kernel_size=1, padding=0, stride=1, groups=1, bias=True), nn.PixelShuffle(2))
        self.conv_l1_k2 = nn.Conv2d(in_channels=120, out_channels=60, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.conv_l1_k3 = NAFBlock(60)

        self.up_l1 = nn.Sequential(nn.Conv2d(in_channels=60, out_channels=120, kernel_size=1, padding=0, stride=1, groups=1, bias=True), nn.PixelShuffle(2))
        self.conv_l0_k2 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=5, padding=2, stride=1, groups=1, bias=True)
        self.conv_l0_k3 = NAFBlock(30)

        self.out_conv = nn.Conv2d(in_channels=30, out_channels=3, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, wien_decon_4x, wien_decon_2x, wien_decon_1x):

        side = (self.params['network_width'] - self.params['out_width']) // 2
        deconv0 = wien_decon_1x[:, :, side:-side, side:-side]
        deconv1 = wien_decon_2x[:, :, side//2:-side//2, side//2:-side//2]
        deconv2 = wien_decon_4x[:, :, side//4:-side//4, side//4:-side//4]

        # Decoder
        conv_l0_k0 = self.conv_l0_k0(deconv0)

        conv_l0_k1 = self.conv_l0_k1(conv_l0_k0)
        down_l0 = self.down_l0(conv_l0_k1)

        conv_l1_k0 = torch.cat([deconv1, down_l0], dim=1)
        conv_l1_k1 = self.conv_l1_k1(conv_l1_k0)
        down_l1 = self.down_l1(conv_l1_k1)

        conv_l2_k0 = torch.cat([deconv2, down_l1], dim=1)
        conv_l2_k1 = self.conv_l2_k1(conv_l2_k0)
        conv_l2_k2 = self.conv_l2_k2(torch.cat([conv_l2_k0, conv_l2_k1], dim=1))
        conv_l2_k3 = self.conv_l2_k3(conv_l2_k2)

        up_l2 = self.up_l2(conv_l2_k3)
        conv_l1_k2 = self.conv_l1_k2(torch.cat([conv_l1_k1, up_l2], dim=1))
        conv_l1_k3 = self.conv_l1_k3(conv_l1_k2)

        up_l1 = self.up_l1(conv_l1_k3)
        conv_l0_k2 = self.conv_l0_k2(torch.cat([conv_l0_k1, up_l1], dim=1))
        conv_l0_k3 = self.conv_l0_k3(conv_l0_k2)

        # output
        out = self.out_conv(conv_l0_k3)
        out = torch.clamp(out, 0.0, 1.0)

        return out

class MFWDFNet(nn.Module):
    def __init__(self, feat_extract, deconv_wnr, fusion_net):
        super(MFWDFNet, self).__init__()

        self.feat_extract = feat_extract
        self.deconv_wnr = deconv_wnr
        self.fusion_net = fusion_net

    def forward(self, inputs):

        img_fp_4x, img_fp_2x, img_fp_1x = self.feat_extract(inputs)
        wien_decon_4x, wien_decon_2x, wien_decon_1x = self.deconv_wnr(img_fp_4x, img_fp_2x, img_fp_1x)
        out = self.fusion_net(wien_decon_4x, wien_decon_2x, wien_decon_1x)

        return out
