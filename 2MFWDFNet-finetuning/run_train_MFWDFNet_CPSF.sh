# Data loading
TRAIN_DIR=./Dataset_sim/dataset_new/
TEST_DIR=./Dataset_sim/eval_data_new/
COLOR_FLAG=True

# Saving and logging
SAVE_DIR=./MFWDFNet_CPSF_ckpt/
LOG_FREQ=800
SAVE_FREQ=400
CKPT_DIR=None
MAX_TO_KEEP=5

# Loss
LOSS_MODE=L2
BATCH_WEIGHTS=1.0
NORM_LOSS_WEIGHT=1.0
P_LOSS_WEIGHT=0.005
SPATIAL_LOSS_WEIGHT=0.0

# FP network structure
WIENER_NUM_PSF=multi_psf

# Training
STEPS=800
ETA_MIN=5E-5
AUG_ROTATE=False

# Optimization
G_LR=0.0001
G_ITERS=1
G_NETWORK=MFWDFNet
SNR_OPT=True
SNR_INIT=4.0

python train.py --train_dir $TRAIN_DIR --test_dir $TEST_DIR --color_flag $COLOR_FLAG --save_dir $SAVE_DIR --log_freq $LOG_FREQ --save_freq $SAVE_FREQ --ckpt_dir $CKPT_DIR --max_to_keep $MAX_TO_KEEP --loss_mode $LOSS_MODE --batch_weights $BATCH_WEIGHTS --Norm_loss_weight $NORM_LOSS_WEIGHT --P_loss_weight $P_LOSS_WEIGHT --Spatial_loss_weight $SPATIAL_LOSS_WEIGHT --wiener_num_psf $WIENER_NUM_PSF --steps $STEPS  --eta_min $ETA_MIN --aug_rotate $AUG_ROTATE --G_lr $G_LR --G_iters $G_ITERS --G_network $G_NETWORK --snr_opt $SNR_OPT --snr_init $SNR_INIT > MFWDFNet_CPSF.txt 2>&1 &
