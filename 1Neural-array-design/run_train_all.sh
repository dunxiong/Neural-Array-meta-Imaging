# Data loading
TRAIN_DIR=../Dataset/dataset_new/
TEST_DIR=../Dataset/eval_data_new/
ARRAY_POSITION_MAT_FILENAME=init_surface_temp300_orgin.mat

# Saving and logging
SAVE_DIR=./real_lens
LOG_FREQ=200
SAVE_FREQ=400
CKPT_DIR=./real_lens
MAX_TO_KEEP=5
LOAD_ALL_VARIABLE=True

# Loss
LOSS_MODE=L1
NORM_LOSS_WEIGHT=0.86
P_LOSS_WEIGHT=0.0
SPATIAL_LOSS_WEIGHT=0.0
MSSSIM_LOSS_WEIGHT=0.14
VGG_LAYERS=block2_conv2,block3_conv2


# Training
EPOCHS=7
POSITION_TRAINFLAG=True
LENS_TRAINFLAG=False
GPU_FLAG=0
POSITION_SCALE=1e3


# Convolution
PSF_MODE=SIM_PSF
LENS_TYPE=Lens1
MAXIMUM_DIA_HALF=60e-6


# Sensor
A_POISSON=0.000055
B_SQRT=0.001
COLOR_FLAG=True
INTEG_SPECTRAL=False

# Optimization
C_DECAY_STEP=80000
C_LR=1e-4
C_LR_END=1e-10
G_DECAY_STEP=80000
G_LR=1e-4
G_LR_END=1e-10
GAMMA_OPT=True
GAMMA_INIT=0.01

nohup python -u train_all.py --train_dir $TRAIN_DIR --test_dir $TEST_DIR  ----position_scale $POSITION_SCALE --array_position_mat_filename $ARRAY_POSITION_MAT_FILENAME --save_dir $SAVE_DIR --log_freq $LOG_FREQ --save_freq $SAVE_FREQ --ckpt_dir $CKPT_DIR --max_to_keep $MAX_TO_KEEP --load_all_variable $LOAD_ALL_VARIABLE --loss_mode $LOSS_MODE --Norm_loss_weight $NORM_LOSS_WEIGHT --P_loss_weight $P_LOSS_WEIGHT --Spatial_loss_weight $SPATIAL_LOSS_WEIGHT  --msssim_loss_weight $MSSSIM_LOSS_WEIGHT --vgg_layers $VGG_LAYERS --epochs $EPOCHS --position_trainflag $POSITION_TRAINFLAG --lens_trainflag $LENS_TRAINFLAG --psf_mode $PSF_MODE --lens_type $LENS_TYPE --maximum_Dia_half $MAXIMUM_DIA_HALF --a_poisson $A_POISSON --b_sqrt $B_SQRT --color_flag $COLOR_FLAG --integ_spectral $INTEG_SPECTRAL --C_decay_step $C_DECAY_STEP --C_lr $C_LR --C_lr_end $C_LR_END --G_decay_step $G_DECAY_STEP --G_lr $G_LR --G_lr_end $G_LR_END  --gamma_opt $GAMMA_OPT --gamma_init $GAMMA_INIT --gpu_flag $GPU_FLAG > log_lens.txt 2>&1 &
