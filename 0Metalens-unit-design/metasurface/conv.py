# Convolution and Fourier operations

import tensorflow as tf
import time

def fft(img):
    img = tf.transpose(tf.cast(img, dtype = tf.complex64), perm = [0, 3, 1, 2])
    Fimg = tf.signal.fft2d(img)
    return Fimg

def ifft(Fimg):
    img = tf.cast(tf.abs(tf.signal.ifft2d(Fimg)), dtype=tf.float32)
    img = tf.transpose(img, perm = [0, 2, 3, 1])
    return img

def psf2otf(psf, h, w):
    psf = tf.image.resize_with_crop_or_pad(psf, h, w)
    psf = tf.transpose(tf.cast(psf, dtype = tf.complex64), perm = [0, 3, 1, 2])
    psf = tf.signal.fftshift(psf, axes=(2,3))
    otf = tf.signal.fft2d(psf)
    return otf


# Forward pass
def convolution_tf(params, args):
    def conv_fn(image, psf):
        if args.conv_mode == 'REAL':
            return image
        assert((image.shape[1]) == params['load_width'])
        assert((image.shape[2]) == params['load_width'])
        otf  = psf2otf(psf, params['load_width'], params['load_width'])
        blur = ifft(fft(image) * otf)
        blur = tf.image.resize_with_crop_or_pad(blur, params['network_width'], params['network_width'])
        return blur
    return conv_fn


# Backwards pass
def deconvolution_tf(params, args):
    def deconv_fn(blur, psf, gamma, G, training):
        start = time.time()
        G_img,G_debug = G([blur, gamma, psf], training=training)
        end = time.time()
        t = end - start         
        return t, G_img, G_debug
    return deconv_fn
