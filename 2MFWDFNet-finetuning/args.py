# Argument parameters file

import argparse

def parse_args():
    def str2bool(v):
        assert(v == 'True' or v == 'False')
        return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser = argparse.ArgumentParser(description='Parameter settings for end-to-end optimization of neural nano-optics')
   
    # Data loading arguments
    parser.add_argument('--train_dir'  , type=str, default='./Neural_array_SIM_Dataset/dataset_new/',  help='Directory of training input images')
    parser.add_argument('--test_dir'   , type=str, default='./Neural_array_SIM_Dataset/eval_data_new', help='Directory of testing input images')
    parser.add_argument('--color_flag', type=str2bool, default=True,  help='color image or not')

    # Saving and logging arguments
    parser.add_argument('--save_dir'   , type=str, default='./MFWDFNet', help='Directory for saving ckpts and TensorBoard file')
    parser.add_argument('--save_freq'  , type=int, default=400, help='Interval to save model')
    parser.add_argument('--log_freq'   , type=int, default=800, help='Interval to write to TensorBoard')
    parser.add_argument('--ckpt_dir'   , type=none_or_str, default='./MFWDFNet', help='Restoring from a checkpoint')
    parser.add_argument('--max_to_keep', type=int, default=5, help='Number of checkpoints to save')

    # Loss arguments
    parser.add_argument('--loss_mode'          , type=str, default='L2')
    parser.add_argument('--batch_weights'      , type=str, default='1.0')
    parser.add_argument('--Norm_loss_weight'   , type=float, default=1.0)
    parser.add_argument('--P_loss_weight'      , type=float, default=0.005)
    parser.add_argument('--Spatial_loss_weight', type=float, default=0.0)

    # FP network structure
    parser.add_argument('--wiener_num_psf'          , type=str, default='multi_psf')
    parser.add_argument('--mode_multi_cpsf', type=str, default='psf_spatial_opt')

    # Training arguments
    parser.add_argument('--steps'     , type=int, default=800, help='Total number of optimization cycles')
    parser.add_argument('--eta_min'   , type=float, default=5e-5, help='CosineAnnealingLR eta_min')
    parser.add_argument('--aug_rotate', type=str2bool, default=False, help='True to rotate PSF during training')

    # Sensor arguments
    parser.add_argument('--a_poisson', type=float, default=0.000055, help='Poisson noise component')
    parser.add_argument('--b_sqrt'   , type=float, default=0.001, help='Gaussian noise standard deviation')

    # Optimization arguments
    parser.add_argument('--G_iters'    , type=int, default=1, help='Number of deconvolution optimization iterations per cycle')
    parser.add_argument('--G_lr'       , type=float, default=1e-4, help='Deconvolution learning rate')
    parser.add_argument('--G_beta1'    , type=float, default=0.9, help='Deconvolution beta1 term for Adam optimizer')
    parser.add_argument('--G_network'  , type=str, default='MFWDFNet', help='Select deconvolution method')
    parser.add_argument('--snr_opt'    , type=str2bool, default=True, help='True to optimize SNR parameter')
    parser.add_argument('--snr_init'   , type=float, default=4.0, help='Initial value of SNR parameter')

    args = parser.parse_args()
    print(args)

    return args
