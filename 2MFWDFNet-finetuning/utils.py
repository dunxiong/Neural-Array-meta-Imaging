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

def deconvolve_wnr(blur, snr, otf):
    
    blur_debug = blur
    otf_conj = torch.conj(otf)
    otf_abs2 = torch.abs(otf) ** 2
    gamma = 1/snr
    wiener_filter = otf_conj / (otf_abs2.to(torch.complex64) + gamma.to(torch.complex64))

    output_fft = wiener_filter * fft(blur)
    output = torch.abs(torch.fft.ifft2(output_fft)).float()
    return output, blur_debug

def sensor_noise(input_layer, params, clip=(1E-20, 1.0)):
    device = input_layer.device
    # Apply Poisson noise
    if params['a_poisson'] > 0:
        a_poisson = torch.tensor(params['a_poisson'], device=device, dtype=torch.float32)

        # Clip the input to avoid extremely small values
        input_layer = torch.clamp(input_layer, min=clip[0], max=100.0)
        
        # Generate Poisson noise
        poisson_rate = input_layer / a_poisson
        sampled = torch.poisson(poisson_rate)  # Poisson sampling
        output = sampled * a_poisson
    else:
        output = input_layer

    # Add Gaussian readout noise
    gauss_noise = torch.normal(mean=0.0, std=params['b_sqrt'], size=output.shape, device=device, dtype=torch.float32)
    output = output + gauss_noise

    # Clipping the final output
    output = torch.clamp(output, min=clip[0], max=clip[1])
    return output