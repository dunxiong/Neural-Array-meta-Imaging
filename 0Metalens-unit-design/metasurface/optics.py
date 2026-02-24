import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import numpy as np
from scipy import interpolate


# Initializes parameters used in the simulation and optimization.
def initialize_params(args):

  # Define the `params` dictionary.
  params = dict({})

  # Units and tensor dimensions.
  params['nanometers'] = 1E-9

  theta_base = args.theta_base   #Field angles
  phi_base = 0.0 # Phi angle for full field simulation. Currently unused.  我理解是y视场角

  # Upsampling for Fourier optics propagation
  params['upsample']      = 1
  params['normalize_psf'] = args.normalize_psf

  # Sensor parameters
  params['sensor_pixel']  = 1.92E-6        # Meters
  params['image_width'] = 864*2
  params['a_poisson']     = args.a_poisson # Poisson noise component
  params['b_sqrt']        = args.b_sqrt    # Gaussian noise standard deviation

  # Focal length
  params['f'] = 4E-3

  # Tensor shape parameters and upsampling.
  lambda_base = [700.0, 650.0, 600.0, 550.0, 500.0, 450.0, 405.0]
  params['lambda_base'] = lambda_base # Screen wavelength
  params['num_lambda'] = np.size(lambda_base)
  params['theta_base'] = theta_base
  params['phi_base'] = phi_base
  params['num_field'] = np.size(theta_base)*np.size(phi_base)

  # PSF grid shape.
  # dim is set to work with the offset PSF training scheme
  if args.offset:
      dim = int(2 * (np.size(params['theta_base']) - 1) - 1)
  else:
      dim = 5

  psfs_grid_shape = [dim, dim]
  params['psfs_grid_shape'] = psfs_grid_shape

  if args.conv == 'patch_size':
      # Patch sized image for training efficiency
      params['psf_width'] = (params['image_width'] // dim)
      assert(params['psf_width'] % 2 == 0)
      params['hw'] = (params['psf_width']) // 2
      params['load_width'] = (params['image_width'] // params['psfs_grid_shape'][0]) + 2*params['psf_width']
      params['network_width'] = (params['image_width'] // params['psfs_grid_shape'][0]) + params['psf_width']
      params['out_width'] = (params['image_width'] // params['psfs_grid_shape'][0])
  elif args.conv == 'full_size':
      # Full size image for inference
      params['psf_width'] = (params['image_width'] // 2)
      print(params['psf_width'])
      assert(params['psf_width'] % 2 == 0)
      params['hw'] = (params['psf_width']) // 2
      params['load_width'] = params['image_width'] + 2*params['psf_width']
      params['network_width'] = params['image_width'] + params['psf_width']
      params['out_width'] = params['image_width']
  else:
    assert 0
      
  print('Image width: {}'.format(params['image_width']))
  print('PSF width: {}'.format(params['psf_width']))
  print('Load width: {}'.format(params['load_width']))
  print('Network width: {}'.format(params['network_width']))
  print('Out width: {}'.format(params['out_width']))

  params['batchSize'] =  np.size(theta_base) * np.size(phi_base)
  batchSize = params['batchSize']
  num_pixels = 520  # Needed for 0.5 mm diameter aperture   num_pixels*params['Lx']
  params['scale'] = 0.6
  params['pixels_aperture'] = num_pixels
  pixelsX = num_pixels
  pixelsY = num_pixels
  params['pixelsX'] = pixelsX
  params['pixelsY'] = pixelsY

  # Simulation grid.
  params['wavelength_nominal'] = 550E-9
  params['pitch'] = 0.32E-6
  params['Lx'] = 3 * params['pitch']
  params['Ly'] = params['Lx']
  dx = params['Lx'] # grid resolution along x
  dy = params['Ly'] # grid resolution along x
  xa = np.linspace(0, pixelsX - 1, pixelsX) * dx # x axis array
  xa = xa - np.mean(xa) # center x axis at zero
  ya = np.linspace(0, pixelsY - 1, pixelsY) * dy # y axis vector
  ya = ya - np.mean(ya) # center y axis at zero
  [y_mesh, x_mesh] = np.meshgrid(ya, xa)
  params['x_mesh'] = x_mesh
  params['y_mesh'] = y_mesh

  # Wavelengths and field angles.
  lam0 = params['nanometers'] * tf.convert_to_tensor(lambda_base, dtype = tf.float32)
  params['q_tensor'] = set_sensor_curve(lam0, color_flag=True)
  lam0 = lam0[tf.newaxis, tf.newaxis, tf.newaxis,:]
  params['lam0'] = lam0

  theta = tf.convert_to_tensor(np.repeat(theta_base, np.size(phi_base)), dtype = tf.float32)
  theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis]
  params['theta'] = theta

  phi = tf.convert_to_tensor(np.tile(phi_base, np.size(theta_base)), dtype = tf.float32)
  phi = phi[:, tf.newaxis, tf.newaxis, tf.newaxis]
  params['phi'] = phi


  # Metasurface proxy phase model.
  params['phase_to_structure_coeffs'] = [0.075303903971995,0.416410588021648,0.321063577213524]
  params['structure_to_phase_coeffs'] = [3.495572,-13.835339,5.4319582,10.91962,-4.9067469,-0.624524]
  params['use_proxy_phase'] = True

  # Compute the PSFs on the full field grid without exploiting azimuthal symmetry.
  params['full_field'] = False  # Not currently used

  # Manufacturing considerations.
  params['fab_tolerancing'] = True #True
  params['fab_error_global'] = 0.02 # +/- 6% duty cycle variation globally (2*sigma)
  params['fab_error_local'] = 0.01 # +/- 3% duty cycle variation locally (2*sigma)

  return params

# Propagate the specified fields to the sensor plane.
def lsasm_propagate(field, params):
  # Field has dimensions of (field, pixelsX, pixelsY, channel)

  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  dx = params['Lx']
  dy = params['Ly']
  z = params['f']

  D = pixelsX * dx

  thetaX = params['theta']
  thetaY = params['phi']
  wvl = params['lam0']

  xi = tf.linspace(0., pixelsX - 1, pixelsX) * dx 
  xi = xi - tf.reduce_mean(xi) 
  eta = tf.linspace(0., pixelsY - 1, pixelsY) * dy 
  eta = eta - tf.reduce_mean(eta) 

  xi = xi[tf.newaxis,:,tf.newaxis,tf.newaxis]
  eta = eta[tf.newaxis,tf.newaxis,:,tf.newaxis]

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

  LRfx = 1200
  LRfy = 1200
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

  return out

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

# Generates a phase distribution modelling a metasurface given some phase coefficients.
def metasurface_phase_generator(phase_coeffs, params):
  x_mesh = params['x_mesh']
  y_mesh = params['y_mesh']
  x_mesh = x_mesh[np.newaxis, :, :, np.newaxis]
  y_mesh = y_mesh[np.newaxis, :, :, np.newaxis]
  phase_def = tf.zeros(shape = np.shape(x_mesh), dtype = tf.float32)
  r_phase = np.sqrt(x_mesh ** 2 + y_mesh ** 2) / (params['pixels_aperture'] * params['Lx'] / 2.0)
  xita = np.arctan2(y_mesh,x_mesh)

  for j in range(np.size(phase_coeffs.numpy())-1):
    power = tf.constant(2 * (j + 1), dtype =  tf.float32)
    r_power = tf.math.pow(r_phase, power)
    phase_def = phase_def + phase_coeffs[j] * r_power
  cubic_base = tf.math.pow(r_phase*tf.cos(xita), 3) + tf.math.pow(r_phase*tf.sin(xita), 3)
  
  cubic_base = tf.cast(cubic_base,dtype =  tf.float32)
  phase_def = phase_def + phase_coeffs[-1] * cubic_base

  phase_def = tf.math.floormod(phase_def, 2 * np.pi)
  if params['use_proxy_phase'] == True:
    # Determine the duty cycle distribution first.
    duty = duty_cycle_from_phase(phase_def, params)

    # Accounts for global and local process variations in grating duty cycle.
    if params['fab_tolerancing'] == True:
      global_error = tf.random.normal(shape = [1], mean = 0.0, stddev = params['fab_error_global'], dtype = tf.float32)
      local_error = tf.random.normal(shape = tf.shape(duty), mean = 0.0, stddev = params['fab_error_local'], dtype = tf.float32)
      duty = duty + global_error + local_error

      # Duty cycle is fit to this range and querying outside is not physically meaningful so we need to clip it.
      duty = tf.clip_by_value(duty, clip_value_min = 0.3, clip_value_max = 0.82)     ####还是不要直接截断，梯度会消失

    phase_def = phase_from_duty_and_lambda(duty, params)
  else:
    phase_def = phase_def * params['wavelength_nominal'] / params['lam0']

  mask = ((x_mesh ** 2 + y_mesh ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)

  phase_def = phase_def * mask
  return phase_def, duty*mask, aperture_layer(duty, scale = params['scale'])

def aperture_layer(input_field, scale = 1):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2,
             -input_shape[2] // 2: input_shape[2] // 2].astype(np.float64)
    max_val = scale * np.amax(x)
    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float32)
    return tf.cast(aperture,dtype = input_field.dtype) * input_field

# Calculates the required duty cycle distribution at the nominal wavelength given
# a specified phase function using a pre-fit polynomial proxy for the mapping.
def duty_cycle_from_phase(phase, params):
  phase = phase / (2 * np.pi)
  p = params['phase_to_structure_coeffs']
  return p[0] * phase ** 2 + p[1] * phase + p[2]

# Calculates the phase shift for a distribution of diameters at all the desired
# simulation wavelengths using a pre-fit polynomial proxy for the mapping.
def phase_from_duty_and_lambda(duty, params):
  p = params['structure_to_phase_coeffs']
  lam = params['lam0'] *1e6
  phase = p[0] + p[1]*lam + p[2]*duty + p[3]*lam**2 + p[4]*lam*duty + p[5]*duty**2
  return phase * 2 * np.pi

# Finds the intensity at the sensor given the input fields.
def compute_intensity_at_sensor(field, params):
  coherent_psf = lsasm_propagate(field, params)
  return tf.math.abs(coherent_psf) ** 2

# Determines the PSF from the intensity at the sensor, accounting for image magnification.
def calculate_psf(intensity, params):

  sensor_pixel = params['sensor_pixel']

  period = params['Lx']

  # Determine PSF shape after optical magnification
  downsampling_scale = int(sensor_pixel/period)
  mag_intensity = area_downsampling_tf(intensity, downsampling_scale)

  # Crop to sensor dimensions
  sensor_psf = mag_intensity
  return sensor_psf

def area_downsampling_tf(input_image, downsampling_scale):
    output_img = tf.nn.avg_pool(input_image,
                            [1, downsampling_scale, downsampling_scale, 1],
                                strides=[1, downsampling_scale, downsampling_scale, 1],
                                padding="VALID")
    return output_img

# Defines a metasurface, including phase and amplitude variation.
def define_metasurface(phase_var, params):
  phase_def, duty, duty_scale = metasurface_phase_generator(phase_var, params)
  phase_def = tf.cast(phase_def, dtype = tf.complex64)
  amp = ((params['x_mesh'] ** 2 + params['y_mesh'] ** 2) < (params['pixels_aperture'] * params['Lx'] / 2.0) ** 2)
  amp = amp.astype(np.float32)
  amp = amp[np.newaxis,:,:,np.newaxis]
  I = 1.0 / np.sum(amp)
  E_amp = np.sqrt(I)
  return amp * E_amp * tf.exp(1j * phase_def), duty,duty_scale

# sensor sample 
def sensor_sample_layer( psf, params):
    
    _, h, w, _ = psf.shape

    q_tensor = params['q_tensor']
    reponse_weight = tf.convert_to_tensor(q_tensor,dtype=tf.float32)
    
    psfs = tf.nn.conv2d(psf,reponse_weight,strides=[1,1,1,1],padding = 'SAME')

    psfs = psfs[:, h // 2 - params['hw'] : h // 2 + params['hw'],
                                                 w // 2 - params['hw'] : w // 2 + params['hw'], :]
    
    if params['normalize_psf']:
      psfs_sum = tf.math.reduce_sum(psfs, axis = (1, 2), keepdims = True)
      psfs = psfs / psfs_sum

    return psfs

def set_sensor_curve(wave_lengths, color_flag = True):
    
    if color_flag:     
          q_r = get_respose_curve(wavelengths = wave_lengths, name = 'QE_R.txt')
          q_g = get_respose_curve(wavelengths = wave_lengths, name = 'QE_G.txt')
          q_b = get_respose_curve(wavelengths = wave_lengths, name = 'QE_B.txt')

          q_r_tensor = np.reshape(q_r, (1,1,len(q_r),1))
          q_g_tensor = np.reshape(q_g, (1,1,len(q_g),1))
          q_b_tensor = np.reshape(q_b, (1,1,len(q_b),1))

          q_tensor = np.concatenate((q_r_tensor, q_g_tensor, q_b_tensor), axis=3) # 读jpg (rgb) 和 mat 文件时图形 波长顺序是反的
    else:
          q_mono = get_respose_curve(wavelengths = wave_lengths, name = 'QE_mono.txt')
          q_tensor = np.reshape(q_mono, (1,1,len(q_mono),1))

    return q_tensor

def get_respose_curve(wavelengths, name = 'QE_R.txt'):
    _wavelengths = wavelengths *1e9
    tmp = np.loadtxt(name)
    f_linear = interpolate.interp1d(tmp[:,0],tmp[:,1])
    qutm_efficience = f_linear(_wavelengths)
    return qutm_efficience/np.sum(qutm_efficience)

# Rotate PSF (non-SVOLA)
def rotate_psfs(psf, params, rotate=True):
  #psfs_grid_shape = params['psfs_grid_shape']
  #rotations = np.zeros(np.prod(psfs_grid_shape))
  psfs = sensor_sample_layer( psf, params)
  rot_angle = 0.0
  if rotate:
    angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0], dtype=np.float32)
    rot_angle = (np.random.choice(angles) * np.pi / 180.0).astype(np.float32)
  rot_angles = tf.fill([np.size(params['theta_base']) * np.size(params['phi_base'])], rot_angle)
  psfs_rot = tfa.image.rotate(psfs, angles = rot_angles, interpolation = 'NEAREST')
  return psfs_rot


def get_psfs(phase_var, params, conv_mode, aug_rotate):
  metasurface_mask,duty,duty_scale = define_metasurface(phase_var, params)
  metasurface_mask = aperture_layer(metasurface_mask, scale = params['scale'])
  intensity = compute_intensity_at_sensor(metasurface_mask, params)
  psf = calculate_psf(intensity, params)
  _, h, w, _ = psf.shape
  psfs = psf[:, h // 2 - params['hw'] : h // 2 + params['hw'], w // 2 - params['hw'] : w // 2 + params['hw'], :]
  psfs_single = sensor_sample_layer( psf, params) #rotate_psfs(psf, params, rotate=False)
  psfs_conv = psfs_single #rotate_psfs(psf, params, rotate=aug_rotate)

  return psfs, psfs_single, psfs_conv,duty,duty_scale


# Applies Poisson noise and adds Gaussian noise.
def sensor_noise_another(input_layer, params, clip = (1E-20,1.)):


  noise_sigma = tf.random.uniform(minval=params['b_sqrt'], maxval=np.sqrt(params['a_poisson']+tf.square(params['b_sqrt'])), shape=[])

  # Add Gaussian readout noise.
  gauss_noise = tf.random.normal(shape=tf.shape(input_layer), mean = 0.0, stddev = noise_sigma, dtype = tf.float32)
  output = input_layer + gauss_noise

  # Clipping.
  output = tf.clip_by_value(output, clip[0], clip[1])
  return output

# Samples wavelengths from a random normal distribution centered about the peak
# wavelengths in the spectra based on the FWHM of each peak.
def randomize_wavelengths(params, lambda_base, sigma):
  pixelsX = params['pixelsX']
  pixelsY = params['pixelsY']
  thetas = params['theta_base']
  phis = params['phi_base']
  lambdas = np.random.normal(lambda_base, sigma)
  lam0 = params['nanometers'] * tf.convert_to_tensor(np.repeat(lambdas, np.size(thetas) * np.size(phis)), dtype = tf.float32)
  lam0 = lam0[:, tf.newaxis, tf.newaxis]
  lam0 = tf.tile(lam0, multiples = (1, pixelsX, pixelsY))
  params['lam0'] = lam0
