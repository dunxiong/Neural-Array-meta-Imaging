import tensorflow as tf
import numpy as np

from util.loss import G_loss
import util.layer as layer
import util.dataset as dataset

import scipy.io as sio


def initialize_params(args):

  # Define the `params` dictionary.
  params = dict({})

  # Meta_parameters
  params['X_pixels_number'] = 1600
  params['Y_pixels_number'] = 1600
  params['Lx'] = 1.725e-6
  params['Ly'] = 1.725e-6
  init_filename = args.array_position_mat_filename   
  position_volume_temp = sio.loadmat(init_filename)
  position_volume_value = position_volume_temp['position'].astype(np.float64)
  position_x_init = position_volume_value[0,:]
  params['array_number'] = np.size(position_x_init)
  params['array_position_mat_filename'] = init_filename
  params['scale'] = args.position_scale

  # Optical parameters
  params['f'] = np.array([4e-3])
  params['wave_lengths'] = np.array([650.,550.,450.]) * 1e-9
  params['wave_lengths_channel_number'] = params['wave_lengths'].size
  params['source_distance'] = np.array([3.])
  
  theta_base = 0.0   #Field angles
  phi_base = 0.0 # Phi angle for full field simulation. Currently unused.  我理解是y视场角
  theta = tf.convert_to_tensor(np.repeat(theta_base, np.size(phi_base)), dtype = tf.float32)
  theta = theta[:, tf.newaxis, tf.newaxis, tf.newaxis]
  params['theta'] = theta

  phi = tf.convert_to_tensor(np.tile(phi_base, np.size(theta_base)), dtype = tf.float32)
  phi = phi[:, tf.newaxis, tf.newaxis, tf.newaxis]
  params['phi'] = phi


  if args.lens_type == 'Lens1':
       params['lens_coief_inits'] = np.array([-86.03,9.576])*1e-2
       params['meta_type'] = 'structure_dispersion'
       params['lens_type'] = 'obtained_by_continual_surface_optimized'

  elif args.lens_type == 'Lens2':
       params['lens_coief_inits'] = np.array([-25451.6147672531,115094.097543125,-215440.262744889,
                                             215877.170098343,-125132.990845656,42312.5944229671,-7978.68880436141,
                                             749.730486490350,-58.2086970067650,-0.561129559320873])*1e-2
       params['meta_type'] = 'structure_dispersion'
       params['lens_type'] = 'obtained_by_random_surface_optimized'

  elif args.lens_type == 'Lens_polar':
       params['lens_coief_inits'] = np.array([-12228.7476065842,56765.3517592509,-110928.060438256,
                                             119035.596293594,-76836.8793393109,30719.4497749759,-7516.69072936418,
                                            1071.98426935990,-113.292663039270,-1.81935835286140e-08])*1e-2						
       params['meta_type'] = 'no_structure_dispersion'
       params['lens_type'] = 'obtained_by_random_surface_optimized'

  elif args.lens_type == 'ideal_lens':
       params['lens_coief_inits'] = np.array([4e-3])*1e-2						
       params['meta_type'] = 'well_achromatic'
       params['lens_type'] = 'ideal_lens'
       params['maximum_Dia_half'] = args.maximum_Dia_half
  else:
       assert False, ("Unsupported meta_type") 
 
  # Sensor parameters
  params['sensor_pixel']  = 6.9e-6        # Meters
  params['sensor_height'] = 512           # Sensor pixels
  params['sensor_width']  = 512           # Sensor pixels
  params['a_poisson']     = args.a_poisson # Poisson noise component
  params['b_sqrt']        = args.b_sqrt    # Gaussian noise standard deviation
  if args.integ_spectral is True:
     params['q_tensor'] = layer.set_sensor_curve(params['wave_lengths'], color_flag=args.color_flag)
  else:
     params['q_tensor'] = None
  params['color'] = args.color_flag
  params['PSF_downsampling_scale'] = np.uint8(params['sensor_pixel'] / params['Lx'])
  params['sensor_channel_number'] = 3

  # Sim parameters
  if args.psf_mode == 'SIM_PSF':
     params['psf_width'] = (params['X_pixels_number'] / params['PSF_downsampling_scale']).astype(np.int32)
  elif args.psf_mode == 'REAL_PSF':
     params['psf_width'] = np.int32(440)

  params['load_width'] = (params['sensor_height'] + params['psf_width']).astype(np.int32)
  params['image_width'] = params['load_width']
  params['network_width'] = params['load_width']
  params['out_width'] = params['sensor_height']
      
  print('Image width: {}'.format(params['image_width']))
  print('PSF width: {}'.format(params['psf_width']))
  print('Load width: {}'.format(params['load_width']))
  print('Network width: {}'.format(params['network_width']))
  print('Out width: {}'.format(params['out_width']))

  params['batchSize'] = 2

  return params

##################################################################
def optical_model_tf(params, args):
    input_image = tf.keras.layers.Input(
        shape=[params['load_width'], params['load_width'], params['wave_lengths_channel_number']], batch_size=None)
    range_volume = tf.keras.layers.Input(shape=[params['X_pixels_number'] * params['Y_pixels_number'] * 9],
                                         batch_size=1)

    if args.psf_mode == 'SIM_PSF':
        input_field1 = layer.set_source_field(params)
        input_field2, height_map2D = layer.array_lens_layer(params, position_train_able=args.position_trainflag,
                                                            coief_train_able=args.lens_trainflag)(range_volume)
        input_field = input_field1 * input_field2
        input_field = layer.aperture_layer(input_field, scale=1)
        psf = layer.lsasm_propagate(input_field, params)

    elif args.psf_mode == 'REAL_PSF':
        psf_temp = sio.loadmat(args.real_psf)
        real_psf = psf_temp['psf_final'].astype(np.float32)
        real_psf = real_psf[np.newaxis, :, :, :]
        real_psf = tf.constant(real_psf, dtype=tf.float32)
        real_psf = tf.image.resize_with_crop_or_pad(real_psf, params['psf_width'], params['psf_width'])
        psf = real_psf / tf.reduce_sum(real_psf, axis=(1, 2), keepdims=True)
    else:
        assert False, ("Unsupported PSF mode")

    conv_img, psfs, GT_img = layer.sensor_sample_layer_tf(params, input_image, psf)
    sensor_img = layer.sensor_noise_another(conv_img,params)

    return tf.keras.Model(inputs=[input_image, range_volume],
                          outputs=[sensor_img, conv_img, GT_img, psfs, psf, height_map2D])


def algorithm_model_tf(params, args):
    input_image = tf.keras.layers.Input(
        shape=[params['network_width'], params['network_width'], params['sensor_channel_number']], batch_size=None)
    input_psf = tf.keras.layers.Input(shape=[params['psf_width'], params['psf_width'], params['sensor_channel_number']],
                                      batch_size=None)

    output_img1 = layer.deconvolve_wnr_tf(args)(input_image, input_psf)
    output_img1 = tf.image.resize_with_crop_or_pad(output_img1, params['out_width'], params['out_width'])
    output_img = layer.neural_network1(output_img1, params)
    return tf.keras.Model(inputs=[input_image, input_psf],
                          outputs=[output_img, output_img1])


##################################################################

## Logging for TensorBoard
def log(name, img, range_volume, C, G, learning_rate_G, learning_rate_C, vgg_model, summary_writer,step, params, args):

    _, conv_img, GT_img, psfs, psf, height_map2D = C([img, range_volume], training=False)
    
    sensor_img = layer.sensor_noise(params,conv_img)

    GT_img = tf.image.resize_with_crop_or_pad(GT_img, params['out_width'], params['out_width'])

    output_img, output_img1 = G([sensor_img, psfs], training=False)

    
    if params['color'] is False:# 灰度图像使用VGG需变成三个通道
         tile_times = tf.constant([1,1,1,3],tf.int32)
         GT_img = tf.tile(GT_img,tile_times)
         output_img = tf.tile(output_img,tile_times)

    # 将输出值规整到 0到1
    GT_img = tf.clip_by_value(GT_img, 0.0, 1.0)
    output_img = tf.clip_by_value(output_img, 0.0, 1.0)
    output_img1 = tf.clip_by_value(output_img1, 0.0, 1.0)

    G_Content_loss_val, G_loss_components, G_metrics = G_loss(output_img, GT_img, vgg_model, args)

    # Save records to TensorBoard
    with summary_writer.as_default():
        # Images
        gamma_vars = G.trainable_variables
        tf.summary.scalar(name = name+'gamma', data = gamma_vars[0][0] ** 2, step=step)

        tf.summary.scalar(name = name+'learnrate_C', data = learning_rate_C(step), step=step)
        tf.summary.scalar(name = name+'learnrate_G', data = learning_rate_G(step), step=step)

        num_patches = params['batchSize']
        for i in range(num_patches):
            tf.summary.image(name = name+'Input/Input_'+str(i), data=GT_img[i:i+1,:,:,:], step=step)
            tf.summary.image(name = name+'Sensor_img/Sensor_img_'+str(i), data=sensor_img[i:i+1,:,:,:], step=step)
            tf.summary.image(name = name+'Output/Output_'+str(i), data=output_img[i:i+1,:,:,:], step=step)
            tf.summary.image(name=name + 'Output1/Output1_' + str(i), data=output_img1[i:i + 1, :, :, :], step=step)

        for i in range(params['wave_lengths_channel_number']):
            tf.summary.image(name = name+'Psf/Psf_wvl_'+str(i), data=psf[:,:,:,i:i+1] / tf.reduce_max(psf[:,:,:,i:i+1],axis=[1,2],keepdims=True), step=step)
            psf_temp = tf.math.log(psf[:,:,:,i:i+1] / tf.reduce_max(psf[:,:,:,i:i+1],axis=[1,2],keepdims=True)+1e-18)
            psf_scale = tf.reduce_max(psf_temp,axis=[1,2],keepdims=True) - tf.reduce_min(psf_temp,axis=[1,2],keepdims=True)
            psf_temp = tf.math.divide((psf_temp - tf.reduce_min(psf_temp,axis=[1,2],keepdims=True)), psf_scale)   
            tf.summary.image(name = name+'Psf/log_Psf_wvl_'+str(i), data=psf_temp, step=step)

        tf.summary.image(name = name+'Psf/Psf', data=psfs / tf.reduce_max(psfs,axis=[1,2,3],keepdims=True), step=step)

        psf_temp = tf.math.log(psfs / tf.reduce_max(psfs,axis=[1,2,3],keepdims=True)+1e-12)
        psf_scale = tf.reduce_max(psf_temp,axis=[1,2],keepdims=True) - tf.reduce_min(psf_temp,axis=[1,2],keepdims=True)
        psf_temp = tf.math.divide((psf_temp - tf.reduce_min(psf_temp,axis=[1,2],keepdims=True)), psf_scale) 
        tf.summary.image(name = name+'Psf/log_Psf', data=psf_temp, step=step)

        tf.summary.image(name = name+'Height_map/Height_map', data= height_map2D / tf.reduce_max(height_map2D,axis=[1,2],keepdims=True), step=step)
        
        # Metrics
        tf.summary.scalar(name = name+'metrics/G_PSNR', data = G_metrics['PSNR'], step=step)
        tf.summary.scalar(name = name+'metrics/G_SSIM', data = G_metrics['SSIM'], step=step)
        tf.summary.scalar(name=name + 'metrics/G_MSSSIM', data=G_metrics['MSSSIM'], step=step)

        # Content losses
        tf.summary.scalar(name = name+'loss/G_Content_loss', data = G_Content_loss_val, step=step)
        tf.summary.scalar(name = name+'loss/G_Norm_loss'   , data = G_loss_components['Norm'], step=step)
        tf.summary.scalar(name = name+'loss/G_P_loss'      , data = G_loss_components['P'], step=step)
        tf.summary.scalar(name = name+'loss/G_Spatial_loss', data = G_loss_components['Spatial'], step=step)
        tf.summary.scalar(name=name + 'loss/G_MSSSIM_loss', data=G_loss_components['MSSSIM'], step=step)
        

def train_step(img, range_volume, C, C_optimizer, G, G_optimizer, vgg_model, params, args):
    
    with tf.GradientTape(persistent = True) as G_tape:   # 一个G_tape用多次！

        if args.position_trainflag:
            sensor_img, _, GT_img, psfs, _, _ = C([img, range_volume], training = True)
        else:
            sensor_img, _, GT_img, psfs, _, _ = C([img, range_volume], training = False)

        GT_img = tf.image.resize_with_crop_or_pad(GT_img, params['out_width'], params['out_width'])

        #sensor_img = layer.sensor_noise(params, sensor_img)

        output_img, _ = G([sensor_img, psfs], training=True)

        if params['color'] is False:# 灰度图像使用VGG需变成三个通道
            tile_times = tf.constant([1,1,1,3],tf.int32)
            GT_img = tf.tile(GT_img,tile_times)
            output_img = tf.tile(output_img,tile_times)

        # 将输出值规整到 0到1
        GT_img = tf.clip_by_value(GT_img, 0.0, 1.0)
        output_img = tf.clip_by_value(output_img, 0.0, 1.0)

        G_loss_all, _, _ = G_loss(output_img, GT_img, vgg_model, args)  

    # Apply gradients
    G_vars = G.trainable_variables
    G_gradients = G_tape.gradient(G_loss_all, G_vars)
    G_optimizer.apply_gradients(zip(G_gradients, G_vars))

    if args.position_trainflag:
       C_vars = C.trainable_variables
       C_gradients = G_tape.gradient(G_loss_all, C_vars)
       C_optimizer.apply_gradients(zip(C_gradients, C_vars))

    del G_tape  #记得del掉！！

    # 给变量赋值用assign

    return G_loss_all


## Training loop
def train_func(args):

    ## init params
    params = initialize_params(args)

    ## build model

    G = algorithm_model_tf(params,args)
    learning_rate_fn_G = tf.keras.optimizers.schedules.PolynomialDecay(args.G_lr, args.G_decay_step, end_learning_rate=1e-10, power=1.0,cycle=False)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_G, beta_1=args.G_beta1)

    C = optical_model_tf(params, args)
    learning_rate_fn_C = tf.keras.optimizers.schedules.PolynomialDecay(args.C_lr, args.C_decay_step, end_learning_rate=args.C_lr_end, power=1.0,cycle=False)
    C_optimizer = tf.keras.optimizers.Adam(learning_rate_fn_C, beta_1=args.C_beta1)

    ## Construct vgg for perceptual loss
    if not args.P_loss_weight == 0:
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg_layers = [vgg.get_layer(name).output for name in args.vgg_layers.split(',')]
        vgg_model = tf.keras.Model(inputs=vgg.input, outputs=vgg_layers)
        vgg_model.trainable = False
    else:
        vgg_model = None

    ## Saving the model 
    checkpoint = tf.train.Checkpoint(G=G,C=C)
    part_checkpoint = tf.train.Checkpoint(G=G)

    print(G.trainable_variables[0])

    max_to_keep = args.max_to_keep
    if args.max_to_keep == 0:
        max_to_keep = None
    manager = tf.train.CheckpointManager(checkpoint, directory=args.save_dir, max_to_keep=max_to_keep)
    ## Loading pre-trained model if exists
    if not args.ckpt_dir == None:
        
        if args.load_all_variable:
            status = checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir, latest_filename=None))
            status.expect_partial() # Silence warnings
            print('load ckpt')
        else:
            status = part_checkpoint.restore(tf.train.latest_checkpoint(args.ckpt_dir, latest_filename=None))
            status.expect_partial() # Silence warnings

    ## Create summary writer for TensorBoard
    summary_writer = tf.summary.create_file_writer(args.logdir_summary_ckpt)
    
    ## Dataset
    
    train_ds = iter(dataset.train_dataset_sim(params['load_width'], args))
    test_ds  = list(dataset.test_dataset_sim(params['load_width'], args).take(1))

    step = 0
    Total_loss = np.zeros((args.step,1))


    init_filename = args.array_position_mat_filename   
    position_volume_temp = sio.loadmat(init_filename)
    range_volume_value = position_volume_temp['Range'].astype(np.float32)
    position_map = range_volume_value.reshape([-1])
    position_map = tf.reshape(position_map, [1, params['X_pixels_number']*params['X_pixels_number']*9])

    for epoch in range(args.epochs):

        #training
        for ind in range(args.step):   # 1365
          
            if step % args.save_freq == 0:
               print('Saving', flush=True)
               manager.save()

            img_train = next(train_ds)

            if step % args.log_freq == 0:
                print('Logging', flush=True)
                log('train_',img_train, position_map, C, G, learning_rate_fn_G, learning_rate_fn_C, vgg_model, summary_writer,step, params, args)
                
                img_test = test_ds[0]
                log('test_',img_test, position_map, C, G, learning_rate_fn_G, learning_rate_fn_C, vgg_model, summary_writer,step, params, args) 


            G_loss_all = train_step(img_train, position_map, C, C_optimizer, G, G_optimizer, vgg_model, params, args)

            Total_loss[ind] = G_loss_all.numpy()

            line = "epoch %d  step %d\n    total_loss %0.8f   \n" % \
                                      (epoch, step, np.mean(Total_loss[np.where(Total_loss)]))
            print(line)
                
            step += 1

    if args.position_trainflag:
        C_vars = C.trainable_variables
        x_position = C_vars[0].numpy() * 1e-3
        y_position = C_vars[1].numpy() * 1e-3
        xita = C_vars[2].numpy() * 1e2
        
    else:
        x_position = 0.
        y_position = 0.
        xita = 0.

    return x_position, y_position, xita