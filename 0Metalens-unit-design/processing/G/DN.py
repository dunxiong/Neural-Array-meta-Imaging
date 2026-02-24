import tensorflow as tf
import numpy as np
import scipy.io as sio
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from numpy.fft import ifftshift

def compl_exp_tf(phase, dtype_complex=tf.complex64, name='complex_exp'):
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype_complex),
                   1.j * tf.cast(tf.sin(phase), dtype=dtype_complex),
                  name=name)

def fft(img):
    img = tf.transpose(tf.cast(img, dtype = tf.complex64), perm = [0, 3, 1, 2])
    Fimg = tf.signal.fft2d(img)
    return Fimg

def ifft(Fimg):
    img = tf.cast(tf.math.abs(tf.signal.ifft2d(Fimg)), dtype=tf.float32)
    img = tf.transpose(img, perm = [0, 2, 3, 1])
    return img

def psf2otf(psf, h, w):
    psf = tf.image.resize_with_crop_or_pad(psf, h, w)
    psf = tf.transpose(tf.cast(psf, dtype = tf.complex64), perm = [0, 3, 1, 2])
    psf = tf.signal.fftshift(psf, axes=(2,3))
    otf = tf.signal.fft2d(psf)
    return otf

def deconvolve_wnr(blur, estimate, psf, gamma):
    pad_width = blur.shape.as_list()[1] // 2
    blur = tf.pad(blur, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
    estimate = tf.pad(estimate, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]]) 
    img_shape = blur.shape.as_list()    
    otf = psf2otf(psf, img_shape[1],img_shape[2])
    wiener_filter = tf.math.conj(otf) / (tf.cast(tf.abs(otf) ** 2, tf.complex64) + tf.cast(gamma, tf.complex64))
    output = tf.cast(tf.math.abs(ifft(wiener_filter * (fft(blur)+fft(estimate)*tf.cast(tf.abs(gamma), tf.complex64)))), tf.float32)
    return output

def conv(filters, size, stride, activation, apply_instnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, stride, padding='same', use_bias=True))
    if apply_instnorm:
        result.add(tfa.layers.InstanceNormalization())
    if not activation == None:
        result.add(activation())
    return result

def conv_transp(filters, size, stride, activation, apply_instnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', use_bias=True))
    if not activation == None:
        result.add(activation())
    return result

def MaxPool2d(pool_size):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.MaxPool2D(pool_size, padding='same'))
    return result

 
def DN(params, args):
    LReLU = tf.keras.layers.LeakyReLU

    h = params['network_width']
    w = params['network_width']

    inputs = tf.keras.layers.Input(shape=[h   ,w   ,3])  
    gamma    = tf.keras.layers.Input(shape=[1])
    psf = tf.keras.layers.Input(shape=[params['psf_width'] ,params['psf_width'],3])
    
    deconv0 = deconvolve_wnr(inputs, inputs, psf, gamma)
    deconv0 = tf.image.resize_with_crop_or_pad(deconv0, params['out_width'], params['out_width'])  

    conv_d1_k0 = conv(32, 3, 1, LReLU, apply_instnorm=False)(deconv0)
    conv_d1_k1 = conv(32, 3, 1, LReLU, apply_instnorm=False)(conv_d1_k0)
    conv_d1_k2 = MaxPool2d((2, 2))(conv_d1_k1)

    conv_d2_k0 = conv(64, 3, 1, LReLU, apply_instnorm=False)(conv_d1_k2)
    conv_d2_k1 = conv(64, 3, 1, LReLU, apply_instnorm=False)(conv_d2_k0)
    conv_d2_k2 = MaxPool2d((2, 2))(conv_d2_k1)

    conv_d3_k0 = conv(128, 3, 1, LReLU, apply_instnorm=False)(conv_d2_k2)
    conv_d3_k1 = conv(128, 3, 1, LReLU, apply_instnorm=False)(conv_d3_k0)
    conv_d3_k2 = MaxPool2d((2, 2))(conv_d3_k1)

    conv_d4_k0 = conv(256, 3, 1, LReLU, apply_instnorm=False)(conv_d3_k2)
    conv_d4_k1 = conv(256, 3, 1, LReLU, apply_instnorm=False)(conv_d4_k0)
    conv_d4_k2 = MaxPool2d((2, 2))(conv_d4_k1)

    conv_b1_k0 = conv(512, 3, 1, LReLU, apply_instnorm=False)(conv_d4_k2)
    conv_b1_k1 = conv(512, 3, 1, LReLU, apply_instnorm=False)(conv_b1_k0)
   
    conv_u1_k1 = conv_transp(256, 3, 1, LReLU, apply_instnorm=False)(conv_b1_k1)
    conv_u1_k2 = conv(256, 3, 1, LReLU, apply_instnorm=False)(tf.concat([conv_d4_k1, conv_u1_k1], axis=3))
    conv_u1_k3 = conv(256, 3, 1, LReLU, apply_instnorm=False)(conv_u1_k2)

    conv_u2_k1 = conv_transp(128, 3, 1, LReLU, apply_instnorm=False)(conv_u1_k3)
    conv_u2_k2 = conv(128, 3, 1, LReLU, apply_instnorm=False)(tf.concat([conv_d3_k1, conv_u2_k1], axis=3))
    conv_u2_k3 = conv(128, 3, 1, LReLU, apply_instnorm=False)(conv_u2_k2)

    conv_u3_k1 = conv_transp(64, 3, 1, LReLU, apply_instnorm=False)(conv_u2_k3)
    conv_u3_k2 = conv(64, 3, 1, LReLU, apply_instnorm=False)(tf.concat([conv_d2_k1, conv_u3_k1], axis=3))
    conv_u3_k3 = conv(64, 3, 1, LReLU, apply_instnorm=False)(conv_u3_k2)

    conv_u4_k1 = conv_transp(32, 3, 1, LReLU, apply_instnorm=False)(conv_u3_k3)
    conv_u4_k2 = conv(32, 3, 1, LReLU, apply_instnorm=False)(tf.concat([conv_d1_k1, conv_u4_k1], axis=3))
    conv_u4_k3 = conv(32, 3, 1, LReLU, apply_instnorm=False)(conv_u4_k2)

    out_temp = conv(3, 3, 1, None, apply_instnorm=False)(conv_u4_k3)
    out = deconv0 + out_temp
    #out = tf.clip_by_value(out, 0.0, 1.0)

    return tf.keras.Model(inputs=[inputs,gamma,psf], 
                          outputs=[out, deconv0])


def DN_no(params, args):
    LReLU = tf.keras.layers.LeakyReLU

    h = params['network_width']
    w = params['network_width']

    inputs = tf.keras.layers.Input(shape=[h   ,w   ,3])  
    gamma    = tf.keras.layers.Input(shape=[1])
    psf = tf.keras.layers.Input(shape=[params['psf_width'] ,params['psf_width'],3])
    
    deconv0 = deconvolve_wnr(inputs, inputs, psf, gamma)
    deconv0 = tf.image.resize_with_crop_or_pad(deconv0, params['out_width'], params['out_width'])  

    return tf.keras.Model(inputs=[inputs,gamma,psf], 
                          outputs=[deconv0,deconv0])


