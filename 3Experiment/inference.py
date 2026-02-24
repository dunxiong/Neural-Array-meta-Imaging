import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from networks.select import select_G
from dataset import train_dataset_sim, test_dataset_sim
from args import parse_args
import scipy.io as sio
from torchvision.models import VGG19_Weights
import os
import cv2
import time

# Initializes parameters used in the simulation and optimization.
def initialize_params(args):

  # Define the `params` dictionary.
  params = dict({})

  params['sensor_height'] = 512           # Sensor pixels
  params['sensor_width']  = 512           # Sensor pixels
  params['psf_width'] = np.int32(440)

  device = 'cuda:0'
  params['device'] = device

  # PSF number when Wiener deconvolove in Feature domain
  if args.wiener_num_psf == 'single_psf':
      psf_dir = './psf.mat'
      psf_temp = sio.loadmat(psf_dir)
      real_psf = psf_temp['psf_final'].astype(np.float32)
      real_psf = torch.tensor(real_psf,device=device).unsqueeze(0)  # (1, h, w, c)
      real_psf = real_psf.permute(0, 3, 1, 2) # (1, c, h, w)
      params['psf'] = real_psf / torch.sum(real_psf, dim=(2, 3), keepdim=True)

  params['image_width'] = params['sensor_height']
  params['load_width'] = 952
  params['network_width'] = params['load_width']
  params['out_width'] = params['sensor_height']
      
  print('Image width: {}'.format(params['image_width']))
  print('PSF width: {}'.format(params['psf_width']))
  print('Load width: {}'.format(params['load_width']))
  print('Network width: {}'.format(params['network_width']))
  print('Out width: {}'.format(params['out_width']))

  return params


## Logging for TensorBoard
def log(img, G, step, params, args):

    with torch.no_grad():

        G.eval()
        
        a = 0.15
        img *= a

        sensor_img = img.to(params['device'])

        # Deconvolution
        start = time.time()
        G_img = G(sensor_img)
        print("Step time: {}\n".format(time.time() - start), flush=True)
        
        G_img = G_img.to('cpu')
        
        G_img = torch.clamp(G_img, 0.0, 1.0)

        G_img = G_img.permute(0, 2, 3, 1)

        G_img = 1.5* G_img[:,241-145:241+155,256-120:256+110,:]

        model_output_temp = np.clip(np.squeeze(G_img.numpy()), 0.0, 1.0)
        model_output_temp = model_output_temp ** (1/1.8)
        
        model_output_temp = model_output_temp * 255

        model_output_temp1 = cv2.cvtColor(model_output_temp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        model_output_file_path = os.path.join(args.ckpt_dir, '%05d.png' % step)
        cv2.imwrite(model_output_file_path, model_output_temp1.astype(np.uint8))

# Training loop
def train(args):
    ## Metasurface
    params = initialize_params(args)

    ## Network architectures
    G = select_G(params, args).to(params['device'])

    # Learning rate scheduler and optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=args.G_lr, betas=(args.G_beta1, 0.999))

    ## Construct VGG for perceptual loss
    if not args.P_loss_weight == 0:
        vgg = torchvision.models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(params['device'])
        layer_indices = {'block2_conv2': 8, 'block3_conv2': 13}
        vgg_layers = 'block2_conv2,block3_conv2'.split(',')
        selected_layer_indices = [layer_indices[name] for name in vgg_layers]
        vgg_features = []
        for _, indices in enumerate(selected_layer_indices):
            vgg_features.append(torch.nn.Sequential(*[vgg.features[i] for i in range(indices + 1)]))

        for param in vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights
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
        checkpoint = torch.load(os.path.join(ckpt_dir, "model_ckpt.pth"),map_location=params['device'])
        G.load_state_dict(checkpoint['G_state_dict'], strict=False)
        # G_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Loading pre-trained model if exists
    if args.ckpt_dir is not None:
        if not os.path.exists(args.ckpt_dir):
            print(f"Checkpoint file not found: {args.ckpt_dir}")
            os.makedirs(args.ckpt_dir, exist_ok=True)
        else:
            load_checkpoint(args.ckpt_dir)

    ## Dataset
    test_ds =  test_dataset_sim(params['load_width'], args)


    for (i, data) in enumerate(test_ds):

        log(data, G, i, params, args)

## Entry point
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    train(args)

if __name__ == '__main__':
    main()
