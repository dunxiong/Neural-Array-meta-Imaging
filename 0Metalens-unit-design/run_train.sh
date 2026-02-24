# Data loading
TRAIN_DIR=../Dataset/dataset_new/
TEST_DIR=../Dataset//eval_data_new/

# Saving and logging
SAVE_DIR=./ckpt
LOG_FREQ=200
SAVE_FREQ=500
CKPT_DIR=./ckpt
MAX_TO_KEEP=10

# Loss
LOSS_MODE=L1
BATCH_WEIGHTS=1.0,1.0,1.0,1.0,1.0
NORM_LOSS_WEIGHT=1.0
P_LOSS_WEIGHT=0.005
VGG_LAYERS=block2_conv2,block3_conv2
SPATIAL_LOSS_WEIGHT=0.0

# Training
AUG_ROTATE=False

# Convolution
PSF_MODE=SIM_PSF
CONV_MODE=SIM
CONV=patch_size
OFFSET=True
NORMALIZE_PSF=True
THETA_BASE=0.0,7.0,12.0,16.0,20.0,24.0

BOUND_VAL=100.0

# Sensor
A_POISSON=0.000055
B_SQRT=0.001

# Optimization
PHASE_LR=1e-5
PHASE_ITERS=1
G_LR=1e-4
G_ITERS=1
G_NETWORK=NoDNN
GAMMA_OPT=True
GAMMA_INIT=0.5

python train.py --train_dir $TRAIN_DIR --test_dir $TEST_DIR --save_dir $SAVE_DIR --log_freq $LOG_FREQ --save_freq $SAVE_FREQ --ckpt_dir $CKPT_DIR --max_to_keep $MAX_TO_KEEP --loss_mode $LOSS_MODE --batch_weights $BATCH_WEIGHTS --Norm_loss_weight $NORM_LOSS_WEIGHT --P_loss_weight $P_LOSS_WEIGHT --vgg_layers $VGG_LAYERS --Spatial_loss_weight $SPATIAL_LOSS_WEIGHT --aug_rotate $AUG_ROTATE --psf_mode $PSF_MODE --conv_mode $CONV_MODE --conv $CONV  --offset $OFFSET --normalize_psf $NORMALIZE_PSF --theta_base $THETA_BASE  --bound_val $BOUND_VAL --a_poisson $A_POISSON --b_sqrt $B_SQRT --Phase_lr $PHASE_LR --Phase_iters $PHASE_ITERS --G_lr $G_LR --G_iters $G_ITERS --G_network $G_NETWORK --gamma_opt $GAMMA_OPT --gamma_init $GAMMA_INIT > lens.txt 2>&1 &
