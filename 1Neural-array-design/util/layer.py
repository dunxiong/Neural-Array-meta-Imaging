import os
import tensorflow as tf
import numpy as np
import scipy.io as sio
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from numpy.fft import ifftshift
from scipy import interpolate


##################################################### bscic function


def compl_exp_tf(phase, dtype_complex=tf.complex64, name='complex_exp'):
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype_complex),
                   1.j * tf.cast(tf.sin(phase), dtype=dtype_complex),
                  name=name)

#######################################################################
'''与原来1.4使用的code一样'''

def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    a_tensor = tf.cast(a_tensor, dtype)
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d_transp = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.transpose(a_fft2d_transp, [0, 2, 3, 1])
    return a_fft2d

def transp_ifft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    a_tensor = tf.cast(a_tensor, dtype)
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor_transp)
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    return a_ifft2d

def fftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def psf2otf_tf(input_filter, output_size, dtype = tf.complex64):

    _, fh, fw,  _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad)
            pad_bottom = pad_right = int(pad)

        padded = tf.pad(input_filter, [[0, 0],[pad_top, pad_bottom],
                                       [pad_left, pad_right],  [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded_temp1 = ifftshift2d_tf(padded)
    otf = transp_fft2d(padded_temp1, dtype=dtype)
    return otf

def img_psf_conv_tf(img, psf, dtype_complex = tf.complex64, dtype_noncomplex = tf.float32):

    img_shape = img.shape.as_list()
    psf_shape = psf.shape.as_list()

    target_side_length = np.maximum(2 * img_shape[1],2*psf_shape[1])

    height_pad = (target_side_length - img_shape[1]) / 2
    width_pad = (target_side_length - img_shape[1]) / 2

    pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
    pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

    img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
    img_shape = img.shape.as_list()

    img_fft = transp_fft2d(img,dtype = dtype_complex)

    otf = psf2otf_tf(psf, output_size=img_shape[1:3],dtype = dtype_complex)

    result_temp = transp_ifft2d(img_fft * otf,dtype = dtype_complex)
    result = tf.cast(tf.math.real(result_temp), dtype = dtype_noncomplex)    ### real or  abs

    result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result


def sensor_sample_layer_tf(params,
                        input_image,
                        psf):
    q_tensor = params['q_tensor']

    if q_tensor is None:

        psfs = psf
        GT_img = input_image
        sensor_img = img_psf_conv_tf(input_image, psf)

    else:
        reponse_weight = tf.convert_to_tensor(q_tensor, dtype=tf.float32)

        GT_img = tf.nn.conv2d(input_image, reponse_weight, strides=[1, 1, 1, 1], padding='SAME')
        psfs = tf.nn.conv2d(psf, reponse_weight, strides=[1, 1, 1, 1], padding='SAME')
        input_image = img_psf_conv_tf(input_image, psf)  # shape of input_image [batch,M,M,channel]
        sensor_img = tf.nn.conv2d(input_image, reponse_weight, strides=[1, 1, 1, 1], padding='SAME')

    return sensor_img, psfs, GT_img


class deconvolve_wnr_tf(tf.keras.layers.Layer):

    def __init__(self, args):
        super(deconvolve_wnr_tf, self).__init__()

        gamma_temp = np.zeros([1],dtype=np.float32)
        gamma_temp[0] = args.gamma_init
        if args.gamma_opt:
           self.gamma0 = tf.Variable(gamma_temp,trainable=True,dtype=tf.float32,name='gamma')
        else:
           self.gamma0 = tf.constant(gamma_temp,dtype=tf.float32)

    def call(self, blur, psf):
        gamma = tf.square(self.gamma0)
        pad_width = blur.shape.as_list()[1] // 2
        blur = tf.pad(blur, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]])
        img_shape = blur.shape.as_list()
        otf = psf2otf_tf(psf, img_shape[1:3])
        wiener_filter = tf.math.conj(otf) / (tf.cast(otf * tf.math.conj(otf), tf.complex64) + tf.cast(gamma, tf.complex64))
        output = tf.cast(tf.math.real(transp_ifft2d(wiener_filter * transp_fft2d(blur))), tf.float32)   #### real or abs
        output = output[:, pad_width:-pad_width, pad_width:-pad_width, :]
        return output


#############################################################################

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



def img_psf_conv(img, psf):

    img_shape = img.shape.as_list()
    psf_shape = psf.shape.as_list()

    target_side_length = np.maximum(2 * img_shape[1],2*psf_shape[1])
    height_pad = (target_side_length - img_shape[1]) / 2
    width_pad = (target_side_length - img_shape[2]) / 2

    pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
    pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

    img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
    img_shape = img.shape.as_list()

    otf = psf2otf(psf, img_shape[1],img_shape[2])

    result = ifft(fft(img) * otf)

    result = result[:,pad_top:-pad_bottom,pad_left:-pad_right,:]
    
    return result


def aperture_layer(input_field, scale = 1):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
             -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)
    max_val = scale * np.amax(x)
    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float32) + 1e-4  ## 用soft边界
    return tf.cast(aperture, dtype = input_field.dtype) * input_field


def get_refractive_idcs_sio2(wavelengths):
    _wavelengths = wavelengths * 1e6
    _refractive_idcs = np.sqrt(1 + 0.6961663*_wavelengths**2/(_wavelengths**2-0.0684043**2) +
                                  0.4079426*_wavelengths**2/(_wavelengths**2-0.1162414**2) +
                                  0.8974794*_wavelengths**2/(_wavelengths**2-9.896161**2))
    return _refractive_idcs


def get_refractive_idcs_pmma(wavelengths):
    _wavelengths = wavelengths * 1e6
    _refractive_idcs = np.sqrt(2.1864582 - 2.4475348e-4*_wavelengths**2 +
                                  1.4144787e-2/_wavelengths**2 -
                                  4.4329781e-4/_wavelengths**4 +
                                  7.7664259e-5/_wavelengths**6 -
                                  2.9936382e-6/_wavelengths**8)
    return _refractive_idcs


def get_respose_curve(wavelengths, name = 'QE_R.txt'):
    _wavelengths = wavelengths *1e9
    tmp = np.loadtxt(name)
    f_linear = interpolate.interp1d(tmp[:,0],tmp[:,1])
    qutm_efficience = f_linear(_wavelengths)
    return qutm_efficience/np.sum(qutm_efficience)


def set_sensor_curve(wave_lengths, color_flag = False):
    
    if color_flag:     
          q_r = get_respose_curve(wavelengths = wave_lengths, name = 'QE_R.txt')
          q_g = get_respose_curve(wavelengths = wave_lengths, name = 'QE_G.txt')
          q_b = get_respose_curve(wavelengths = wave_lengths, name = 'QE_B.txt')

          q_r_tensor = np.reshape(q_r, (1,1,len(q_r),1))
          q_g_tensor = np.reshape(q_g, (1,1,len(q_g),1))
          q_b_tensor = np.reshape(q_b, (1,1,len(q_b),1))

          q_tensor = np.concatenate((q_r_tensor,q_g_tensor,q_b_tensor),axis = 3)
    else:
          q_mono = get_respose_curve(wavelengths = wave_lengths, name = 'QE_mono.txt')
          q_tensor = np.reshape(q_mono, (1,1,len(q_mono),1))

    return q_tensor

def get_intensities(input_field):
    return tf.math.real(input_field * tf.math.conj(input_field))    ### real or abs

def area_downsampling_tf(input_image, downsampling_scale):
    output_img = tf.nn.avg_pool(input_image,
                                [1, downsampling_scale, downsampling_scale, 1],
                                 strides=[1, downsampling_scale, downsampling_scale, 1],
                                 padding="VALID")
    return output_img

def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs - 1.    # shape [1,1,1,n]
    # wave number
    wave_nos = 2. * np.pi / wave_lengths   # shape [1,1,1,n]

    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = compl_exp_tf(phi, dtype = tf.complex64, name='DOE_phase')
    return phase_shifts

###################################################### input field ####################

def set_source_field(params):

    pixelsX = params['X_pixels_number']
    pixelsY = params['Y_pixels_number']

    dx = params['Lx']
    dy = params['Ly']


    [x, y] = np.mgrid[-pixelsX//2:pixelsX//2,
                      -pixelsY//2:pixelsY//2].astype(np.float64)

    x = x * dx
    y = y * dy

    squared_sum = x ** 2 + y ** 2
    squared_sum = squared_sum[np.newaxis,:,:,np.newaxis] 

    wave_lengths = params['wave_lengths']
    wave_lengths = wave_lengths[np.newaxis,np.newaxis,np.newaxis,:]
    wave_nos = 2. * np.pi / wave_lengths

    source_distance = params['source_distance']
    source_distance = source_distance[:,np.newaxis,np.newaxis,np.newaxis]

    curvature = tf.sqrt(squared_sum + tf.cast(source_distance,tf.float64)**2)  

    phase_def =  wave_nos * curvature

    return compl_exp_tf(phase_def,dtype_complex=tf.complex64,name='point_source')

#######################################################optical element layler################
class array_lens_layer(tf.keras.layers.Layer):
  def __init__(self, params, position_train_able=True, xita_train_able=True, coief_train_able=False):
        super(array_lens_layer, self).__init__()

        wave_lengths = params['wave_lengths']
        refractive_idcs = get_refractive_idcs_sio2(wave_lengths)  
        self.wave_lengths = tf.cast(tf.reshape(wave_lengths,[1,1,1,-1]),dtype=tf.float32)
        self.refractive_idcs = tf.cast(tf.reshape(refractive_idcs,[1,1,1,-1]),dtype=tf.float32)

        self.pixelsX = params['X_pixels_number']
        self.pixelsY = params['Y_pixels_number']
        dx = params['Lx']
        dy = params['Ly']
        [x, y] = np.mgrid[0:self.pixelsX,
                          0:self.pixelsY]
        x = x[:,:,np.newaxis]
        y = y[:,:,np.newaxis]
        
        self.xx = tf.constant(x,dtype=tf.int64)
        self.yy = tf.constant(y,dtype=tf.int64)
        
        x = (x+0.5)*dx
        y = (y+0.5)*dy
        self.X = tf.cast(x,dtype=tf.float32)
        self.Y = tf.cast(y,dtype=tf.float32)
        
        init_filename = params['array_position_mat_filename']
        position_volume_temp = sio.loadmat(init_filename)
        position_volume_value = position_volume_temp['position'].astype(np.float64)
        position_x_init = position_volume_value[0,:] * params['scale']
        position_y_init = position_volume_value[1,:] * params['scale']
        position_x_initializer = tf.convert_to_tensor(position_x_init,dtype = tf.float32)
        position_y_initializer = tf.convert_to_tensor(position_y_init,dtype = tf.float32)
        self.position_x0 = tf.Variable(initial_value = position_x_initializer,
                                  trainable=position_train_able,
                                  dtype=tf.float32,
                                  name='position_x')
        self.position_y0 = tf.Variable(initial_value = position_y_initializer,
                                  trainable=position_train_able,
                                  dtype=tf.float32,
                                  name='position_y')
        #### xita
        xita = position_volume_temp['xita_offset'].astype(np.float32)
        xita = xita.reshape(-1) * 1e-2

        xita_initializer = tf.convert_to_tensor(xita,dtype = tf.float32)
        
        self.xita0 = tf.Variable(initial_value = xita_initializer,
                            trainable=xita_train_able,
                            dtype=tf.float32,
                            name='xita')
        #####
        lens_inits = params['lens_coief_inits']
        # lens_inits = lens_inits[:,np.newaxis,np.newaxis,np.newaxis]
        lens_initializer = tf.convert_to_tensor(lens_inits,dtype = tf.float32)
        self.lens_coeffs_temp = tf.Variable(initial_value = lens_initializer,
                                  trainable=coief_train_able,
                                  dtype=tf.float32,
                                  name='lens_coeffs')
                                  
        self.params = params

  def call(self, range_volume):
        position_x = self.position_x0 / self.params['scale']
        position_y = self.position_y0 / self.params['scale']
        xita = self.xita0 * 1e2
        lens_coeffs = self.lens_coeffs_temp*1e2 
    
        range_volume_graph = tf.cast(range_volume, tf.int32)
        center_x = tf.gather(position_x, range_volume_graph, axis=0)
        center_x = tf.reshape(center_x,(self.pixelsX,self.pixelsX,9))
        center_y = tf.gather(position_y, range_volume_graph, axis=0)
        center_y = tf.reshape(center_y,(self.pixelsX,self.pixelsX,9))
        
        xita_offset = tf.gather(xita, range_volume_graph, axis=0)   # 旋转角度
        xita_offset = tf.reshape(xita_offset,(self.pixelsX,self.pixelsX,9))

        xita_orgin = tf.math.angle(tf.complex(self.Y-center_x,self.X-center_y))

        r = tf.square(self.Y-center_x) +tf.square(self.X-center_y)  # matlab 和pyhon 定义的reshape是反的
        r = tf.reduce_min(r,axis = 2,keepdims=True)
        
        position = tf.argmin(r,axis=2)
        position = tf.expand_dims(position,axis=2)
        
        index = tf.concat([self.xx,self.yy,position],axis = 2)
        index = tf.reshape(index,(self.pixelsX*self.pixelsX,3))

        xita_orgin = xita_orgin + tf.cast(xita_offset,dtype=tf.float32)
        xita_final = tf.gather_nd(xita_orgin,index)

        xita_final = tf.reshape(xita_final,(self.pixelsX,self.pixelsX,1))
    
        if self.params['lens_type'] == 'obtained_by_continual_surface_optimized':
            r_final = r * 1.5949e7 
            
            [x, y] = np.mgrid[-self.pixelsX // 2: self.pixelsX // 2,
             -self.pixelsX // 2: self.pixelsX // 2].astype(np.float64)
            r = np.sqrt(x ** 2 + y ** 2)
            max_val = np.amax(x)
            aperture = (r < max_val).astype(np.int8)
            # print(aperture)
            aperture = np.reshape(aperture,-1)
            # print(aperture)
            range = np.where(aperture == 1)
            r_range = tf.reshape(r_final,[self.pixelsX*self.pixelsX])
            r_range_new = tf.gather(r_range, range[0], axis=0)
            
            r_mean = tf.reduce_mean(tf.pow(r_range_new,0.5))
            a_scale = (2/3) / r_mean

            phi = lens_coeffs[0] * r_final + a_scale * lens_coeffs[1] * tf.pow(r_final,1.5) * tf.sin(xita_final) + a_scale * lens_coeffs[1] * tf.pow(r_final,1.5) * tf.cos(xita_final)
            phi = tf.expand_dims(phi,axis = 0)
            
            phase_def = tf.math.floormod(phi, 2 * np.pi)

        elif self.params['lens_type'] == 'obtained_by_random_surface_optimized':
            r = tf.expand_dims(r,axis = 0)
            r = r * 1.5949e7
            r = r - 0.36   # 限制透镜的最大尺寸  我们是0.6的平方
            r = tf.minimum(0.05*r, r)
            r = r + 0.36
            r = r/0.36

            #注意计算精度
            lens_base = tf.concat([tf.pow(r,9),tf.pow(r,8),
                                  tf.pow(r,7),tf.pow(r,6),tf.pow(r,5), tf.pow(r,4), tf.pow(r,3),
                                  tf.pow(r,2),r], axis=0)
            phi = tf.reduce_sum(lens_base * lens_coeffs[0:9,:,:,:], axis = 0, keepdims=True) + lens_coeffs[9:,:,:,:]
            phase_def = tf.math.floormod(phi, 2 * np.pi)

        elif self.params['lens_type'] == 'ideal_lens':
             f = lens_coeffs
             r = r - params['maximum_Dia_half'] * params['maximum_Dia_half'] 
             r = tf.minimum(0.01*r, r)
             r = r + params['maximum_Dia_half'] * params['maximum_Dia_half'] 

             z_height = f-tf.sqrt(f*f+r)
             phase_def = 2. * np.pi / 0.55e-6 * z_height
             
        else:
             assert False, ("Unsupported lens_type")  


        if self.params['meta_type'] == 'structure_dispersion':
            phase_def = phase_def / (2 * np.pi)

            # phase 2 structure
            #params = np.array([0.00190747017083288,0.0662738034394481,0.321063577213524])
            params = np.array([0.075303903971995,0.416410588021648,0.321063577213524])
            structure = params[0] * phase_def ** 2 + params[1] * phase_def + params[2]

            structure = tf.clip_by_value(structure, clip_value_min = 0.3, clip_value_max = 0.82)   # 保险起见

            # structure2 phase
            #params_new = [16.4+5.56332726,-86.93,34.13,68.61,-30.83,-3.924]
            params_new = [3.49557210017397,-13.8353392029785,5.43195820772639,10.9196206455349,-4.90674689552313,-0.624523996692597]
            wave_lengths = self.wave_lengths * 1e6
            phi_all = params_new[0]+params_new[1] * wave_lengths + params_new[2] * structure + params_new[3] * (wave_lengths**2) + params_new[4] * wave_lengths * structure +params_new[5] * (structure**2);

            phi_all = phi_all * (2*np.pi)

        elif self.params['meta_type'] == 'no_structure_dispersion':
            phi_all = phase_def
            structure = phase_def

        elif self.params['meta_type'] == 'well_achromatic':
            wave_nos_ratio = 0.55e-6 / self.wave_lengths
            phi_all = phase_def * wave_nos_ratio
            structure = phase_def - tf.reduce_min(phase_def)
        else:
            assert False, ("Unsupported meta_type") 

        return compl_exp_tf(phi_all), structure

############################# light propogation ###########################

# Propagate the specified fields to the sensor plane.
def lsasm_propagate(field, params):
  # Field has dimensions of (field, pixelsX, pixelsY, channel)

  pixelsX = params['X_pixels_number']
  pixelsY = params['Y_pixels_number']
  dx = params['Lx']
  dy = params['Ly']
  z = params['f']

  D = pixelsX * dx

  wvl = params['wave_lengths']
  wvl = wvl[tf.newaxis, tf.newaxis, tf.newaxis,:]

  thetaX = params['theta']
  thetaY = params['phi']

  xi = tf.linspace(0., pixelsX - 1, pixelsX) * dx 
  xi = xi - tf.reduce_mean(xi) 
  eta = tf.linspace(0., pixelsY - 1, pixelsY) * dy 
  eta = eta - tf.reduce_mean(eta) 

  xi = xi[tf.newaxis,:,tf.newaxis,tf.newaxis]
  eta = eta[tf.newaxis,tf.newaxis,:,tf.newaxis]

#   x = tf.linspace(0., 600 - 1, 600) * 1.725e-6
#   x = x - tf.reduce_mean(x)
#   y = x

#   x = x[tf.newaxis,:,tf.newaxis,tf.newaxis]
#   y = y[tf.newaxis,tf.newaxis,:,tf.newaxis]


  xc = - z * tf.sin(thetaX / 180 * np.pi) / tf.sqrt(1 - tf.sin(thetaX / 180 * np.pi)**2 - tf.sin(thetaY / 180 * np.pi)**2)
  yc = - z * tf.sin(thetaY / 180 * np.pi) / tf.sqrt(1 - tf.sin(thetaX / 180 * np.pi)**2 - tf.sin(thetaY / 180 * np.pi)**2)

  x = xi + xc
  y = eta + yc

  k = 2 * np.pi / wvl 
  
  _,Nx,_,_ = xi.shape
  _,_,Ny,_ = eta.shape
  Lfx = (Nx - 1) / D
  Lfy = (Ny - 1) / D
  
  fcX = - tf.sin(thetaX / 180.0 * np.pi) * k / (2 * np.pi)
  fcY = - tf.sin(thetaY / 180.0 * np.pi) * k / (2 * np.pi)
  offx = fcX
  offy = fcY
  
  # fxmax = Lfx / 2 + tf.abs(offx)
  # fymax = Lfy / 2 + tf.abs(offy)
  # fxmax = tf.clip_by_value(fxmax, -1 / wvl, 1 / wvl)
  # fymax = tf.clip_by_value(fymax, -1 / wvl, 1 / wvl)

  # condition = 1 - (wvl * fxmax)**2 - (wvl * fymax)**2 <= 0
  # eps = 1e-9
  # beta = tf.atan2(fymax, fxmax)
  # fxmax = tf.where(condition, tf.minimum(fxmax, tf.cos(beta) / (wvl + eps)), fxmax)
  # fymax = tf.where(condition, tf.minimum(fymax, tf.sin(beta) / (wvl + eps)), fymax)
  # Lfx = tf.where(condition, (fxmax - tf.abs(offx)) * 2, Lfx)
  # Lfy = tf.where(condition, (fymax - tf.abs(offy)) * 2, Lfy)

  LRfx = 1500
  LRfy = 1500
  dfx2 = Lfx / LRfx
  dfy2 = Lfy / LRfy

  fx = tf.linspace(0., LRfx - 1, LRfx) * dfx2
  fx = fx - tf.reduce_mean(fx)
  fy = tf.linspace(0., LRfy - 1, LRfy) * dfy2
  fy = fy - tf.reduce_mean(fy)

  fxx, fyy = tf.meshgrid(fx, fy, indexing='xy')

  fxx = fxx[tf.newaxis, :, :, tf.newaxis] + offx
  fyy = fyy[tf.newaxis, :, :, tf.newaxis] + offy

  fx_shift = fx[tf.newaxis, :, tf.newaxis, tf.newaxis] + offx
  fy_shift = fy[tf.newaxis, tf.newaxis, :, tf.newaxis] + offy

  H = tf.exp(1j * tf.cast(k * (z * tf.sqrt(1 - (fxx * wvl)**2 - (fyy * wvl)**2)), dtype=tf.complex64))

  Fu = mdft(field, xi, eta, fx_shift - offx, fy_shift - offy)
  out = midft(Fu * H, x, y, fx_shift, fy_shift)

  psf = get_intensities(out)

  downsampling_scale = params['PSF_downsampling_scale']
  psf = area_downsampling_tf(psf, downsampling_scale)    # shape [k,M,M,n]

  return tf.math.divide(psf, tf.math.reduce_sum(psf, axis=[1, 2], keepdims=True), name='psf_depth_idx')

def mdft(in_matrix, x, y, fx, fy):

    in_matrix = tf.transpose(in_matrix, perm = [0,3,1,2])
    x = tf.transpose(x, perm = [0,3,1,2])
    fx = tf.transpose(fx, perm = [0,3,2,1])
    y = tf.transpose(y, perm = [0,3,1,2])
    fy = tf.transpose(fy, perm = [0,3,2,1])

    mx = tf.exp(-2 * np.pi * 1j * tf.cast(tf.matmul(x, fx), dtype=tf.complex64)) 
    my = tf.exp(-2 * np.pi * 1j * tf.cast(tf.matmul(fy, y), dtype=tf.complex64))

    out_matrix = tf.matmul(tf.matmul(my, in_matrix), mx)

    _,_,lx,_ = x.shape
    _,_,_,ly = y.shape
    
    if lx == 1:
        dx = 1
    else:
        dx = (x[:,:,-1,:] - x[:,:,0,:]) / (lx - 1)
        dx = tf.cast(dx[:,:,tf.newaxis,:], dtype=tf.complex64)

    if ly == 1:
        dy = 1
    else:
        dy = (y[:,:,:,-1] - y[:,:,:,0]) / (ly - 1)
        dy = tf.cast(dy[:, :, :, tf.newaxis], dtype=tf.complex64)

    out_matrix = out_matrix * dx * dy  # the result is only valid for uniform sampling
    out_matrix = tf.transpose(out_matrix, [0, 2, 3, 1])

    return out_matrix


def midft(in_matrix, x, y, fx, fy):

    in_matrix = tf.transpose(in_matrix, perm = [0,3,1,2])
    x = tf.transpose(x, perm = [0,3,2,1])
    fx = tf.transpose(fx, perm = [0,3,1,2])
    y = tf.transpose(y, perm = [0,3,2,1])
    fy = tf.transpose(fy, perm = [0,3,1,2])

    mx = tf.exp(2 * np.pi * 1j * tf.cast(tf.matmul(fx, x), dtype=tf.complex64))
    my = tf.exp(2 * np.pi * 1j * tf.cast(tf.matmul(y, fy), dtype=tf.complex64))
    
    out_matrix = tf.matmul(tf.matmul(my, in_matrix), mx)

    _,_,lfx,_ = fx.shape
    _,_,_,lfy = fy.shape
    if lfx == 1:
        dfx = 1
    else:
        dfx = (fx[:,:,-1,:] - fx[:,:,0,:]) / (lfx - 1)
        dfx = tf.cast(dfx[:,:,tf.newaxis,:], dtype=tf.complex64)

    if lfy == 1:
        dfy = 1
    else:
        dfy = (fy[:,:,:,-1] - fy[:,:,:,0]) / (lfy - 1)
        dfy = tf.cast(dfy[:, :, :, tf.newaxis], dtype=tf.complex64)

    out_matrix = out_matrix * dfx * dfy  # the result is only valid for uniform sampling
    out_matrix = tf.transpose(out_matrix, [0, 2, 3, 1])

    return out_matrix

################################################# sensor layer ###############################




def sensor_noise(params, input_layer, clip = (1E-20,1.)):

  # Apply Poisson noise.
  if (params['a_poisson'] > 0):

      a_poisson_tf = tf.constant(params['a_poisson'], dtype = tf.float32)
      input_layer = tf.clip_by_value(input_layer, clip[0], 100.0)
      p = tfp.distributions.Poisson(rate = input_layer / a_poisson_tf, validate_args = True)

      sampled = tfp.monte_carlo.expectation(f = lambda x: x, samples = p.sample(1), log_prob = p.log_prob, use_reparameterization = False)
      output = sampled * a_poisson_tf
  else:
      output = input_layer

  # Add Gaussian readout noise.
  gauss_noise = tf.random.normal(shape=tf.shape(output), mean = 0.0, stddev = params['b_sqrt'], dtype = tf.float32)
  output = output + gauss_noise

  # Clipping.
  output = tf.clip_by_value(output, clip[0], clip[1])
  return output


def sensor_noise_another(input_layer, params, clip = (1E-20,1.)):

  noise_sigma = tf.random.uniform(minval=params['b_sqrt'], maxval=np.sqrt(params['a_poisson']+tf.square(params['b_sqrt'])), shape=[])

  # Add Gaussian readout noise.
  gauss_noise = tf.random.normal(shape=tf.shape(input_layer), mean = 0.0, stddev = noise_sigma, dtype = tf.float32)
  output = input_layer + gauss_noise

  # Clipping.
  output = tf.clip_by_value(output, clip[0], clip[1])
  return output


############################################# algorithm #########################################


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

 
def neural_network1(blur,params):
    LReLU = tf.keras.layers.LeakyReLU

    conv_d1_k0 = conv(32, 3, 1, LReLU, apply_instnorm=False)(blur)
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
    if params['color'] is True:
       out_temp = conv(3, 3, 1, None, apply_instnorm=False)(conv_u4_k3)
    else:
       out_temp = conv(1, 3, 1, None, apply_instnorm=False)(conv_u4_k3) 
    out = blur + out_temp
    return out







