# Argument parameters file
import numpy as np
import argparse

def parse_args():
    def str2bool(v):
        if isinstance(v,bool):
            return v
        if v == 'True':
            return True
        if v == 'False':
            return False
        #assert(v == 'True' or v == 'False')
        #return v.lower() in ('true')

    def none_or_str(value):
        if value.lower() == 'none':
            return None
        return value

    parser = argparse.ArgumentParser(description='Parameter settings for end-to-end optimization of metalens array')

    # Data loading arguments   set/hyperspectral_dataset
    parser.add_argument('--train_dir'  , type=str, default='/home/dx/Dataset/dataset_new/',  help='Directory of training input images')
    parser.add_argument('--test_dir'   , type=str, default='/home/dx/Dataset/eval_data_new/', help='Directory of testing input images')
    parser.add_argument('--array_position_mat_filename', type=str, default='init_surface_temp300_pB.mat', help='Directory of testing input images')
    parser.add_argument('--position_scale', type=float, default=1e3)
    
    # Saving and logging arguments
    parser.add_argument('--save_dir'   , type=str, default = './real_lens120221102', help='Directory for saving ckpts')
    parser.add_argument('--save_summary_dir', type=str, default='./real_lens120221102', help='Directory for saving ckpts and TensorBoard file')
    parser.add_argument('--save_freq'  , type=int, default=400, help='Interval to save model')
    parser.add_argument('--log_freq'   , type=int, default=200, help='Interval to write to TensorBoard')
    parser.add_argument('--ckpt_dir'   , type=none_or_str, default='./real_lens1_20221102', help='Restoring from a checkpoint')
    parser.add_argument('--max_to_keep', type=int, default=5, help='Number of checkpoints to save')
    parser.add_argument('--load_all_variable', type=str2bool, default=False, help='whether to load all variable')

    # Loss arguments
    parser.add_argument('--loss_mode'          , type=str, default='L1')
    parser.add_argument('--Norm_loss_weight'   , type=float, default=0.86)
    parser.add_argument('--P_loss_weight'      , type=float, default=0.0)
    parser.add_argument('--Spatial_loss_weight', type=float, default=0.0)
    parser.add_argument('--msssim_loss_weight', type=float, default=0.14)
    parser.add_argument('--vgg_layers'         , type=str, default='block2_conv2,block3_conv2')

    # Training arguments
    parser.add_argument('--epochs'     , type=int, default=7, help='Total epochs of each step cycle')
    parser.add_argument('--step', type=int, default=1365, help='Total number of each epochs')
    parser.add_argument('--position_trainflag'     , type=str2bool, default=True, help='lens_array is training or not')
    parser.add_argument('--lens_trainflag'     , type=str2bool, default=False, help='lens_params is training or not')
    parser.add_argument('--gpu_flag'     , type=str, default='0', help='which gpu to be used')

    # optical arguments
    parser.add_argument('--psf_mode'     , type=str, default='SIM_PSF', help='Use simulated PSF(SIM_PSF) or captured PSF (REAL_psf)')
    parser.add_argument('--real_psf'     , type=str, default='./psf0810.mat', help='mat of experimentally measured PSF')
    parser.add_argument('--lens_type'     , type=str, default='Lens1', help='mat of experimentally measured PSF')
    parser.add_argument('--maximum_Dia_half'     , type=float, default=60e-6, help='mat of experimentally measured PSF')

    # Sensor arguments
    parser.add_argument('--a_poisson', type=float, default=0.00004, help='Poisson noise component')
    parser.add_argument('--b_sqrt'   , type=float, default=0.00001, help='Gaussian noise standard deviation')
    parser.add_argument('--color_flag'      , type=str2bool, default=True, help='color sensor or not')
    parser.add_argument('--integ_spectral'      , type=str2bool, default=False, help='integ sensor spectral or not')
    
    # Optimization arguments
    parser.add_argument('--optical_network'  , type=str, default='fdtd', help='Select optical method')
    parser.add_argument('--deconv_network'  , type=str, default='neural', help='Select deconv method')
    parser.add_argument('--C_decay_step', type=int, default=80000, help='Number of decay step for learning rate')
    parser.add_argument('--C_lr'   , type=float, default=1e-4, help='Meta-optic learning rate')
    parser.add_argument('--C_lr_end'   , type=float, default=1e-10, help='Meta-optic learning rate end')
    parser.add_argument('--C_beta1', type=float, default=0.9, help='Meta-optic beta1 term for Adam optimizer')
    parser.add_argument('--G_decay_step'    , type=int, default=80000, help='Number of decay step for learning rate')
    parser.add_argument('--G_lr'       , type=float, default=1e-4, help='Deconvolution learning rate')
    parser.add_argument('--G_lr_end', type=float, default=1e-10, help='Deconvolution learning rate end')
    parser.add_argument('--G_beta1'    , type=float, default=0.9, help='Deconvolution beta1 term for Adam optimizer')
    parser.add_argument('--gamma_opt'    , type=str2bool, default=True, help='True to optimize SNR parameter')
    parser.add_argument('--gamma_init'   , type=float, default=np.sqrt(0.0001), help='Initial value of SNR parameter')

    args, unknow = parser.parse_known_args()   #在notebook中应用需要改成这样
    #args = parser.parse_args()
    print(args)
    return args
