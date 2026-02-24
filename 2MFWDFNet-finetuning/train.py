import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from networks.select import select_G
from dataset import train_dataset_sim, test_dataset_sim
from loss import G_loss
from args import parse_args
import scipy.io as sio
from utils import resize_with_crop_or_pad, sensor_noise
from torchvision.models import VGG19_Weights
from torch.utils.tensorboard import SummaryWriter
import os
import time
import h5py

# Initializes parameters used in the simulation and optimization.
def initialize_params(args):

  # Define the `params` dictionary.
  params = dict({})

  params['sensor_pixel']  = 6.9E-6          # Meters
  params['sensor_height'] = 640           # Sensor pixels
  params['sensor_width']  = 640           # Sensor pixels
  params['psf_width'] = np.int32(400)

  torch.cuda.empty_cache()
  device = 'cuda:0'
  params['device'] = device
  
  params['a_poisson'] = args.a_poisson
  params['b_sqrt'] = args.b_sqrt

  # PSF number when Wiener deconvolove in Feature domain
  if args.wiener_num_psf == 'single_psf':
      psf_dir = './single_psf.mat'
      psf_temp = sio.loadmat(psf_dir)
      real_psf = psf_temp['psf_RGB'].astype(np.float32)
      real_psf = torch.tensor(real_psf,device=device).unsqueeze(0)  # (1, h, w, c)
      params['psf'] = real_psf / torch.sum(real_psf, dim=(1, 2), keepdim=True)

  elif args.wiener_num_psf == 'multi_psf':
      psf_dir = './multi_psf.mat'
      psf_temp = sio.loadmat(psf_dir)
      real_psf = psf_temp['psf_new'].astype(np.float32)
      real_psf = np.transpose(real_psf, (2, 0, 1, 3))
      psf_5num = torch.tensor(real_psf[0:5, :, :, :], device=device)
      params['psf'] = psf_5num / torch.sum(psf_5num, dim=(1, 2), keepdim=True)

      psf_all = torch.tensor(real_psf, device=device)
      psf_all = torch.cat([psf_all[i, :, :, :] for i in range(9)], dim=2).unsqueeze(0)  # 27 channel
      params['psf_all'] = psf_all

  params['image_width'] = params['sensor_height']
  params['load_width'] = (params['image_width'] + params['psf_width']).astype(np.int32)
  params['network_width'] = params['load_width']
  params['out_width'] = 480
      
  print('Image width: {}'.format(params['image_width']))
  print('PSF width: {}'.format(params['psf_width']))
  print('Load width: {}'.format(params['load_width']))
  print('Network width: {}'.format(params['network_width']))
  print('Out width: {}'.format(params['out_width']))

  return params


## Logging for TensorBoard
def log(name, img, G, G_optimizer, vgg_model, summary_writer, step, params, args):

    with torch.no_grad():

        G.eval()

        GT_img = img[0].to(params['device'])
        conv_image = img[1].to(params['device'])
        
        sensor_img = sensor_noise(conv_image, params)  # Could add solver.sensor_noise equivalent if needed

        # Deconvolution
        G_img = G(sensor_img)

        # Losses
        gt_img = resize_with_crop_or_pad(GT_img, params['out_width'], params['out_width'])

        G_Content_loss_val, G_loss_components, G_metrics = G_loss(G_img, gt_img, vgg_model, args)

   
        for param_group in G_optimizer.param_groups:
            lr = param_group['lr']
            summary_writer.add_scalar(name+'Learning Rate', lr, global_step=step)

        # snr =  G.deconv_wnr.snr.detach().cpu().numpy()
        
        # Save records to TensorBoard
        # Images
        summary_writer.add_image(name+'Input/GT', gt_img[0,:,:,:], global_step=step)
        num_patches = 1
        for i in range(num_patches):
            summary_writer.add_image(name+'Output/Output_' + str(i), G_img[i, :, :, :], global_step=step)
            summary_writer.add_image(name+'Blur/Blur_' + str(i), conv_image[i, :, :, :], global_step=step)
            summary_writer.add_image(name+'Sensor/Sensor_' + str(i), sensor_img[i, :, :, :], global_step=step)

        # Metrics
        summary_writer.add_scalar(name+'metrics/G_PSNR', G_metrics['PSNR'], global_step=step)
        summary_writer.add_scalar(name+'metrics/G_SSIM', G_metrics['SSIM'], global_step=step)
        # summary_writer.add_scalar('SNR', snr, global_step=step)

        # Content losses
        summary_writer.add_scalar(name+'loss/G_Content_loss', G_Content_loss_val, global_step=step)
        summary_writer.add_scalar(name+'loss/G_Norm_loss', G_loss_components['Norm'], global_step=step)
        summary_writer.add_scalar(name+'loss/G_P_loss', G_loss_components['P'], global_step=step)
        summary_writer.add_scalar(name+'loss/G_Spatial_loss', G_loss_components['Spatial'], global_step=step)
    
## Optimization Step
def train_step(img, G, G_optimizer, vgg_model, step, params, args):

    GT_img = img[0].to(params['device'])

    conv_image = img[1].to(params['device'])
    sensor_img = sensor_noise(conv_image, params)  # Could add solver.sensor_noise equivalent if needed

    G_optimizer.zero_grad()  # Clear previous gradients

    # Forward pass through the model
    G_img = G(sensor_img)

    # Losses
    gt_img = resize_with_crop_or_pad(GT_img, params['out_width'], params['out_width'])
    G_loss_val, G_loss_components, G_metrics = G_loss(G_img, gt_img, vgg_model, args)

    # Backward pass and optimizer step
    G_loss_val.backward()  # Compute gradients
    G_optimizer.step()  # Apply gradients

    return G, G_optimizer, G_loss_val

# Training loop
def train(args):
    ## Metasurface
    params = initialize_params(args)

    ## Network architectures
    G = select_G(params, args).to(params['device'])

    # Learning rate scheduler and optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=args.G_lr, betas=(args.G_beta1, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(G_optimizer, args.steps, args.eta_min)

    ## Construct VGG for perceptual loss
    if not args.P_loss_weight == 0:
        vgg = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).to(params['device'])
        layer_indices = {'block2_conv2': 8, 'block3_conv2': 13}
        vgg_layers = 'block2_conv2,block3_conv2'.split(',')
        selected_layer_indices = [layer_indices[name] for name in vgg_layers]
        vgg_features = []
        for _, indices in enumerate(selected_layer_indices):
            vgg_features.append(torch.nn.Sequential(*[vgg.features[i] for i in range(indices + 1)]))

        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

        tf_conv_layers = ["block1_conv1", "block1_conv2", "block2_conv1", "block2_conv2",
                          "block3_conv1", "block3_conv2", "block3_conv3", "block3_conv4",
                          "block4_conv1", "block4_conv2", "block4_conv3", "block4_conv4",
                          "block5_conv1", "block5_conv2", "block5_conv3", "block5_conv4"]
        for i in range(len(vgg_features)):

            with h5py.File("vgg19_block3_conv2_weights_tf.h5", "r") as f:
                tf_layer_idx = 0  

                for pt_layer in vgg_features[i]:
                    if isinstance(pt_layer, torch.nn.Conv2d):
                        tf_layer_name = tf_conv_layers[tf_layer_idx]
                        tf_layer_idx += 1  

                        weights = f[tf_layer_name][tf_layer_name]["kernel:0"][()]
                        bias = f[tf_layer_name][tf_layer_name]["bias:0"][()]

                        # TensorFlow [H, W, In, Out] -> PyTorch [Out, In, H, W]
                        weights = torch.tensor(weights).permute(3, 2, 0, 1)

                        pt_layer.weight.data = weights.to('cuda:0')
                        pt_layer.bias.data = torch.tensor(bias).to('cuda:0')
    else:
        vgg_features = None

    ## Saving the model
    def save_checkpoint(model, optimizer, save_dir, epoch, save_freq, max_to_keep=5):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_path = os.path.join(save_dir, f'model_checkpoint_{str(int(epoch/save_freq))}.pth')
        torch.save({'G_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),}, checkpoint_path)

        checkpoints = sorted([f for f in os.listdir(save_dir) if f.endswith('.pth')], key=lambda x: os.path.getctime(os.path.join(save_dir, x)))
        if len(checkpoints) > max_to_keep:
            os.remove(os.path.join(save_dir, checkpoints[0]))  
            

    def load_checkpoint(ckpt_dir):
        checkpoint = torch.load(os.path.join(ckpt_dir, "model_checkpoint.pth"),map_location=params['device'])
        G.load_state_dict(checkpoint['G_state_dict'])
        G_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Loading pre-trained model if exists
    if args.ckpt_dir is not None:
        if not os.path.exists(args.ckpt_dir):
            print(f"Checkpoint file not found: {args.ckpt_dir}")
            os.makedirs(args.ckpt_dir, exist_ok=True)
        else:
            load_checkpoint(args.ckpt_dir)

    ## Create summary writer for TensorBoard
    summary_writer = SummaryWriter(log_dir=args.save_dir)

    ## Dataset
    train_ds = train_dataset_sim(params['load_width'], args)
    test_ds =  test_dataset_sim(params['load_width'], args)

    ## Do training
    loop = 0
    best_psnr = 0
    min_loss =1

    for e in range(args.steps):

        print('Epoch:' + str(e + 1) + '/' + str(args.steps))

        G.train()

        for (i, data) in enumerate(train_ds):

            start = time.time()
            gt_img = data[0].to(params['device'])
            conv_image = data[1].to(params['device'])
            sensor_img = sensor_noise(conv_image, params)  # Could add solver.sensor_noise equivalent if needed

            G_optimizer.zero_grad()  # Clear previous gradients

            G_img = G(sensor_img)

            GT_img = resize_with_crop_or_pad(gt_img, params['out_width'], params['out_width'])
            G_loss_val, G_loss_components, G_metrics = G_loss(G_img, GT_img, vgg_features, args)

            G_loss_val.backward()  # Compute gradients
            G_optimizer.step()  # Apply gradients

            # Log results periodically
            if loop % args.log_freq == 0:
                print('Logging', flush=True)
                img = list(test_ds)[0]
                log('test_', img, G, G_optimizer, vgg_features, summary_writer, loop, params, args)
            
            if loop % 200 == 0:
                print('Loop:'+str(loop))
                print(f"Step time: {time.time() - start}", flush=True)
            loop = loop + 1

        scheduler.step()
 
        with torch.no_grad():
            G.eval()
            loss_val = []
            psnr_val = []
        
            for batch, img in enumerate(test_ds, 1):
        
                gt_img = img[0].to(params['device'])
                conv_image = img[1].to(params['device'])
                sensor_img = sensor_noise(conv_image, params)  # Could add solver.sensor_noise equivalent if needed
        
                G_img = G(sensor_img)
                GT_img = resize_with_crop_or_pad(gt_img, params['out_width'], params['out_width'])
                G_loss_val, G_loss_components, G_metrics = G_loss(G_img, GT_img, vgg_features, args)
        
                psnr = 20 * torch.log10(1.0 / (F.mse_loss(G_img, GT_img))**0.5)
        
                loss_val += [G_loss_val.item()]
                psnr_val += [psnr.item()]
        
        if (np.mean(loss_val) < min_loss):
            save_checkpoint(G, G_optimizer, args.save_dir, loop, args.save_freq, args.max_to_keep)
            min_loss = np.mean(loss_val)
            print(f"Best psnr: {np.mean(psnr_val)}", flush=True)
            print("=>saved best model")


## Entry point
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    train(args)

if __name__ == '__main__':
    main()
