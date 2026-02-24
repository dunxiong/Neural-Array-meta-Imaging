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
    parser.add_argument('--test_dir'   , type=str, default='./measurements', help='Directory of testing input images')
    parser.add_argument('--color_flag', type=str2bool, default=True,  help='color image or not')

    # Saving and logging arguments
    parser.add_argument('--save_dir'   , type=str, default='./MFWDFNet_ckpt', help='Directory for saving ckpts and TensorBoard file')
    parser.add_argument('--save_freq'  , type=int, default=400, help='Interval to save model')
    parser.add_argument('--log_freq'   , type=int, default=800, help='Interval to write to TensorBoard')
    parser.add_argument('--ckpt_dir'   , type=none_or_str, default='./MFWDFNet_ckpt/', help='Restoring from a checkpoint')
    parser.add_argument('--max_to_keep', type=int, default=5, help='Number of checkpoints to save')

    # Loss arguments
    parser.add_argument('--loss_mode'          , type=str, default='L2')
    parser.add_argument('--batch_weights'      , type=str, default='1.0')
    parser.add_argument('--Norm_loss_weight'   , type=float, default=1.0)
    parser.add_argument('--P_loss_weight'      , type=float, default=0.0)
    parser.add_argument('--Spatial_loss_weight', type=float, default=0.0)
    # parser.add_argument('--spatial_lossweight_mode' , type=str, default='sp_sigmoid')
    parser.add_argument('--threshold', type=float, default=512)


    # MFWDFNet network structure
    parser.add_argument('--wiener_num_psf'          , type=str, default='single_psf')
    # parser.add_argument('--mode_multi_cpsf', type=str, default='psf_frequency_opt_with_initial')
    parser.add_argument('--fusion_method'          , type=str, default='fusion_nafnet')

    # Training arguments
    parser.add_argument('--steps'     , type=int, default=800, help='Total number of optimization cycles')
    parser.add_argument('--aug_rotate', type=str2bool, default=False, help='True to rotate PSF during training')

    # Optimization arguments
    parser.add_argument('--G_iters'    , type=int, default=1, help='Number of deconvolution optimization iterations per cycle')
    parser.add_argument('--G_lr'       , type=float, default=1e-4, help='Deconvolution learning rate')
    parser.add_argument('--G_beta1'    , type=float, default=0.9, help='Deconvolution beta1 term for Adam optimizer')
    parser.add_argument('--G_network'  , type=str, default='MFWDFNet', help='Select deconvolution method')
    parser.add_argument('--snr_opt'    , type=str2bool, default=True, help='True to optimize SNR parameter')
    parser.add_argument('--snr_init'   , type=float, default=4.36, help='Initial value of SNR parameter')

    args = parser.parse_args()
    print(args)

    return args
