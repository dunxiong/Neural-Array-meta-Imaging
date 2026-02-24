import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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

def deconvolve_wnr(blur, estimate, psf, gamma):

    pad_width = blur.shape[-2] // 2
    blur = F.pad(blur, (pad_width, pad_width, pad_width, pad_width), mode='constant', value=0)
    estimate = F.pad(estimate, (pad_width, pad_width, pad_width, pad_width), mode='constant', value=0)
    otf = psf2otf(psf, blur.shape[-2],blur.shape[-1])
    otf_conj = torch.conj(otf)
    otf_abs2 = torch.abs(otf) ** 2
    wiener_filter = otf_conj / (otf_abs2.to(torch.complex64) + gamma.to(torch.complex64))

    output_fft = wiener_filter * (fft(blur)+fft(estimate)*torch.abs(gamma).to(torch.complex64))
    output = torch.abs(torch.fft.ifft2(output_fft)).float()
    return output


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
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=1, output_padding=1, bias=True)]
        if not activation == None:
            layers.append(activation)
        self.conv_transp = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_transp(x)
    

class WinerU(nn.Module):
    def __init__(self, params, args):
        super(WinerU, self).__init__()
        
        self.LReLU = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.color_flag = args.color_flag
        self.params = params

        in_channels = 3 if self.color_flag else 1

        self.gamma = nn.Parameter(torch.tensor(np.sqrt(1.6e-5)), requires_grad=True)

        real_psf = params['psf']
        self.psf = real_psf.permute(0, 3, 1, 2)  # (1, c, h, w)

        self.conv_d1_k0 = Conv(in_channels, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d1_k1 = Conv(32, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d1_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_d2_k0 = Conv(32, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d2_k1 = Conv(64, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d2_k2 = nn.MaxPool2d(kernel_size=2, padding=0)
        
        self.conv_d3_k0 = Conv(64, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d3_k1 = Conv(128, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d3_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_d4_k0 = Conv(128, 256, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d4_k1 = Conv(256, 256, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d4_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_b1_k0 = Conv(256, 512, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_b1_k1 = Conv(512, 512, 3, 1, self.LReLU, apply_instnorm=False)
        
        self.conv_u1_k1 = ConvTransp(512, 256, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u1_k2 = Conv(512, 256, 3, 1, self.LReLU, apply_instnorm=False)  # Concatenation increases channels
        self.conv_u1_k3 = Conv(256, 256, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u2_k1 = ConvTransp(256, 128, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u2_k2 = Conv(256, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u2_k3 = Conv(128, 128, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u3_k1 = ConvTransp(128, 64, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u3_k2 = Conv(128, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u3_k3 = Conv(64, 64, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u4_k1 = ConvTransp(64, 32, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u4_k2 = Conv(64, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u4_k3 = Conv(32, 32, 3, 1, self.LReLU, apply_instnorm=False)
        
        self.output_conv = Conv(32, 3 if self.color_flag else 1, 3, 1, None, apply_instnorm=False)

    def forward(self, inputs):

        deconv0 = deconvolve_wnr(inputs, inputs, self.psf, self.gamma**2)
        deconv0 = resize_with_crop_or_pad(deconv0, self.params['out_width'], self.params['out_width'])  

        # First downsampling block
        conv_d1_k0 = self.conv_d1_k0(deconv0)
        conv_d1_k1 = self.conv_d1_k1(conv_d1_k0)
        conv_d1_k2 = self.conv_d1_k2(conv_d1_k1)

        # Second downsampling block
        conv_d2_k0 = self.conv_d2_k0(conv_d1_k2)
        conv_d2_k1 = self.conv_d2_k1(conv_d2_k0)
        conv_d2_k2 = self.conv_d2_k2(conv_d2_k1)

        # Third downsampling block
        conv_d3_k0 = self.conv_d3_k0(conv_d2_k2)
        conv_d3_k1 = self.conv_d3_k1(conv_d3_k0)
        conv_d3_k2 = self.conv_d3_k2(conv_d3_k1)

        # Fourth downsampling block
        conv_d4_k0 = self.conv_d4_k0(conv_d3_k2)
        conv_d4_k1 = self.conv_d4_k1(conv_d4_k0)
        conv_d4_k2 = self.conv_d4_k2(conv_d4_k1)

        # Bottom block
        conv_b1_k0 = self.conv_b1_k0(conv_d4_k2)
        conv_b1_k1 = self.conv_b1_k1(conv_b1_k0)

        # First upsampling block
        conv_u1_k1 = self.conv_u1_k1(conv_b1_k1)
        conv_u1_k2 = self.conv_u1_k2(torch.cat([conv_d4_k1, conv_u1_k1], dim=1))
        conv_u1_k3 = self.conv_u1_k3(conv_u1_k2)
        
        # Second upsampling block
        conv_u2_k1 = self.conv_u2_k1(conv_u1_k3)
        conv_u2_k2 = self.conv_u2_k2(torch.cat([conv_d3_k1, conv_u2_k1], dim=1))
        conv_u2_k3 = self.conv_u2_k3(conv_u2_k2)

        # Third upsampling block
        conv_u3_k1 = self.conv_u3_k1(conv_u2_k3)
        conv_u3_k2 = self.conv_u3_k2(torch.cat([conv_d2_k1, conv_u3_k1], dim=1))
        conv_u3_k3 = self.conv_u3_k3(conv_u3_k2)

        # Fourth upsampling block
        conv_u4_k1 = self.conv_u4_k1(conv_u3_k3)
        conv_u4_k2 = self.conv_u4_k2(torch.cat([conv_d1_k1, conv_u4_k1], dim=1))
        conv_u4_k3 = self.conv_u4_k3(conv_u4_k2)

        # Final output
        out_temp = self.output_conv(conv_u4_k3)
        out = deconv0 + out_temp
        
        return out

class MultiWinerU(nn.Module):
    def __init__(self, params, args):
        super(MultiWinerU, self).__init__()

        self.LReLU = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.color_flag = args.color_flag
        self.params = params
        in_channels = 3 if self.color_flag else 1

        self.gamma = nn.Parameter(torch.tensor(np.sqrt(1.6e-5)*np.ones((1,27,1,1))), requires_grad=True)
        
        real_psf = params['psf_all']
        real_psf = real_psf.permute(0, 3, 1, 2)  # (1, c, h, w)
        self.psf = nn.Parameter(torch.sqrt(real_psf), requires_grad=True)

        self.conv_d1_k0 = Conv(in_channels*9, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d1_k1 = Conv(32, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d1_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_d2_k0 = Conv(32, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d2_k1 = Conv(64, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d2_k2 = nn.MaxPool2d(kernel_size=2, padding=0)
        
        self.conv_d3_k0 = Conv(64, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d3_k1 = Conv(128, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d3_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_d4_k0 = Conv(128, 256, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d4_k1 = Conv(256, 256, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d4_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_b1_k0 = Conv(256, 512, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_b1_k1 = Conv(512, 512, 3, 1, self.LReLU, apply_instnorm=False)
        
        self.conv_u1_k1 = ConvTransp(512, 256, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u1_k2 = Conv(512, 256, 3, 1, self.LReLU, apply_instnorm=False)  # Concatenation increases channels
        self.conv_u1_k3 = Conv(256, 256, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u2_k1 = ConvTransp(256, 128, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u2_k2 = Conv(256, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u2_k3 = Conv(128, 128, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u3_k1 = ConvTransp(128, 64, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u3_k2 = Conv(128, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u3_k3 = Conv(64, 64, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u4_k1 = ConvTransp(64, 32, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u4_k2 = Conv(64, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u4_k3 = Conv(32, 32, 3, 1, self.LReLU, apply_instnorm=False)

        self.output_conv = Conv(32, 3 if self.color_flag else 1, 3, 1, None, apply_instnorm=False)

    def forward(self, inputs):

        psf = torch.square(self.psf)
        psf = psf/torch.sum(psf, dim=(2, 3), keepdim=True)
        inputs = inputs.repeat(1, 9, 1, 1)
        deconv0 = deconvolve_wnr(inputs, inputs, psf, self.gamma**2)
        deconv0 = resize_with_crop_or_pad(deconv0, self.params['out_width'], self.params['out_width'])

        # First downsampling block
        conv_d1_k0 = self.conv_d1_k0(deconv0)
        conv_d1_k1 = self.conv_d1_k1(conv_d1_k0)
        conv_d1_k2 = self.conv_d1_k2(conv_d1_k1)

        # Second downsampling block
        conv_d2_k0 = self.conv_d2_k0(conv_d1_k2)
        conv_d2_k1 = self.conv_d2_k1(conv_d2_k0)
        conv_d2_k2 = self.conv_d2_k2(conv_d2_k1)

        # Third downsampling block
        conv_d3_k0 = self.conv_d3_k0(conv_d2_k2)
        conv_d3_k1 = self.conv_d3_k1(conv_d3_k0)
        conv_d3_k2 = self.conv_d3_k2(conv_d3_k1)

        # Fourth downsampling block
        conv_d4_k0 = self.conv_d4_k0(conv_d3_k2)
        conv_d4_k1 = self.conv_d4_k1(conv_d4_k0)
        conv_d4_k2 = self.conv_d4_k2(conv_d4_k1)

        # Bottom block
        conv_b1_k0 = self.conv_b1_k0(conv_d4_k2)
        conv_b1_k1 = self.conv_b1_k1(conv_b1_k0)

        # First upsampling block
        conv_u1_k1 = self.conv_u1_k1(conv_b1_k1)
        conv_u1_k2 = self.conv_u1_k2(torch.cat([conv_d4_k1, conv_u1_k1], dim=1))
        conv_u1_k3 = self.conv_u1_k3(conv_u1_k2)

        # Second upsampling block
        conv_u2_k1 = self.conv_u2_k1(conv_u1_k3)
        conv_u2_k2 = self.conv_u2_k2(torch.cat([conv_d3_k1, conv_u2_k1], dim=1))
        conv_u2_k3 = self.conv_u2_k3(conv_u2_k2)

        # Third upsampling block
        conv_u3_k1 = self.conv_u3_k1(conv_u2_k3)
        conv_u3_k2 = self.conv_u3_k2(torch.cat([conv_d2_k1, conv_u3_k1], dim=1))
        conv_u3_k3 = self.conv_u3_k3(conv_u3_k2)

        # Fourth upsampling block
        conv_u4_k1 = self.conv_u4_k1(conv_u3_k3)
        conv_u4_k2 = self.conv_u4_k2(torch.cat([conv_d1_k1, conv_u4_k1], dim=1))
        conv_u4_k3 = self.conv_u4_k3(conv_u4_k2)

        # Final output
        out_temp = self.output_conv(conv_u4_k3)
        out = out_temp

        return out
    
class MultiFlatNet(nn.Module):
    def __init__(self, params, args):
        super(MultiFlatNet, self).__init__()

        self.LReLU = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.color_flag = args.color_flag
        self.params = params
        in_channels = 3 if self.color_flag else 1

        self.gamma = nn.Parameter(torch.tensor(np.sqrt(1.6e-5)*np.ones((1,27,1,1))), requires_grad=True)
        
        real_psf = params['psf_all']
        real_psf = real_psf.permute(0, 3, 1, 2)  # (1, c, h, w)

        otf = psf2otf(real_psf, params['load_width'], params['load_width'])

        initial_wiener =  torch.conj(otf)/((torch.abs(otf) ** 2).to(torch.complex64) + (1/torch.pow(10.0, torch.tensor(args.snr_init, dtype=torch.float32))).to(torch.complex64))

        self.wiener_re = nn.Parameter(initial_wiener.real, requires_grad=True)
        self.wiener_im = nn.Parameter(initial_wiener.imag, requires_grad=True)
        self.normalizer = nn.Parameter(torch.ones(27, dtype=torch.float32)* 10)

        self.conv_d1_k0 = Conv(in_channels*9, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d1_k1 = Conv(32, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d1_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_d2_k0 = Conv(32, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d2_k1 = Conv(64, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d2_k2 = nn.MaxPool2d(kernel_size=2, padding=0)
        
        self.conv_d3_k0 = Conv(64, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d3_k1 = Conv(128, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d3_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_d4_k0 = Conv(128, 256, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d4_k1 = Conv(256, 256, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_d4_k2 = nn.MaxPool2d(kernel_size=2, padding=0)

        self.conv_b1_k0 = Conv(256, 512, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_b1_k1 = Conv(512, 512, 3, 1, self.LReLU, apply_instnorm=False)
        
        self.conv_u1_k1 = ConvTransp(512, 256, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u1_k2 = Conv(512, 256, 3, 1, self.LReLU, apply_instnorm=False)  # Concatenation increases channels
        self.conv_u1_k3 = Conv(256, 256, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u2_k1 = ConvTransp(256, 128, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u2_k2 = Conv(256, 128, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u2_k3 = Conv(128, 128, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u3_k1 = ConvTransp(128, 64, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u3_k2 = Conv(128, 64, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u3_k3 = Conv(64, 64, 3, 1, self.LReLU, apply_instnorm=False)

        self.conv_u4_k1 = ConvTransp(64, 32, 3, 2, self.LReLU, apply_instnorm=False)
        self.conv_u4_k2 = Conv(64, 32, 3, 1, self.LReLU, apply_instnorm=False)
        self.conv_u4_k3 = Conv(32, 32, 3, 1, self.LReLU, apply_instnorm=False)

        self.output_conv = Conv(32, 3 if self.color_flag else 1, 3, 1, None, apply_instnorm=False)

    def forward(self, inputs):

        inputs = inputs.repeat(1, 9, 1, 1)
        wiener = torch.complex(self.wiener_re, self.wiener_im)
        normalizer = self.normalizer.view(1, 27, 1, 1)
        deconv0 = torch.abs(torch.fft.ifft2(wiener * fft(inputs))*normalizer/10).float()
        deconv0 = resize_with_crop_or_pad(deconv0, self.params['out_width'], self.params['out_width'])

        # First downsampling block
        conv_d1_k0 = self.conv_d1_k0(deconv0)
        conv_d1_k1 = self.conv_d1_k1(conv_d1_k0)
        conv_d1_k2 = self.conv_d1_k2(conv_d1_k1)

        # Second downsampling block
        conv_d2_k0 = self.conv_d2_k0(conv_d1_k2)
        conv_d2_k1 = self.conv_d2_k1(conv_d2_k0)
        conv_d2_k2 = self.conv_d2_k2(conv_d2_k1)

        # Third downsampling block
        conv_d3_k0 = self.conv_d3_k0(conv_d2_k2)
        conv_d3_k1 = self.conv_d3_k1(conv_d3_k0)
        conv_d3_k2 = self.conv_d3_k2(conv_d3_k1)

        # Fourth downsampling block
        conv_d4_k0 = self.conv_d4_k0(conv_d3_k2)
        conv_d4_k1 = self.conv_d4_k1(conv_d4_k0)
        conv_d4_k2 = self.conv_d4_k2(conv_d4_k1)

        # Bottom block
        conv_b1_k0 = self.conv_b1_k0(conv_d4_k2)
        conv_b1_k1 = self.conv_b1_k1(conv_b1_k0)

        # First upsampling block
        conv_u1_k1 = self.conv_u1_k1(conv_b1_k1)
        conv_u1_k2 = self.conv_u1_k2(torch.cat([conv_d4_k1, conv_u1_k1], dim=1))
        conv_u1_k3 = self.conv_u1_k3(conv_u1_k2)

        # Second upsampling block
        conv_u2_k1 = self.conv_u2_k1(conv_u1_k3)
        conv_u2_k2 = self.conv_u2_k2(torch.cat([conv_d3_k1, conv_u2_k1], dim=1))
        conv_u2_k3 = self.conv_u2_k3(conv_u2_k2)

        # Third upsampling block
        conv_u3_k1 = self.conv_u3_k1(conv_u2_k3)
        conv_u3_k2 = self.conv_u3_k2(torch.cat([conv_d2_k1, conv_u3_k1], dim=1))
        conv_u3_k3 = self.conv_u3_k3(conv_u3_k2)

        # Fourth upsampling block
        conv_u4_k1 = self.conv_u4_k1(conv_u3_k3)
        conv_u4_k2 = self.conv_u4_k2(torch.cat([conv_d1_k1, conv_u4_k1], dim=1))
        conv_u4_k3 = self.conv_u4_k3(conv_u4_k2)

        # Final output
        out_temp = self.output_conv(conv_u4_k3)
        out = out_temp

        return out
    