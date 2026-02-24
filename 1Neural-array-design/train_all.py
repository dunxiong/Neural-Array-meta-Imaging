
from util.args import parse_args
from train_func import train_func
import util.position_process as PPS
import os
import numpy as np
import tensorflow as tf

def train_all(args):
    for i in range(8):

        learning_rate = 1e-4 * (8 - i) / 8
        decay_steps = np.int32((8 - i) * args.step*7 + 1)
        if i == 7:
            merge_position_threshold = 80e-6
        else:
            merge_position_threshold = 80e-6 * (1 + i / 4)

        args.logdir_summary_ckpt = args.save_dir + '/step' + str(i)

        args.C_lr = learning_rate
        args.G_lr = learning_rate
        args.C_decay_step = decay_steps
        args.G_decay_step = decay_steps

        if i > 0:
           args.load_all_variable = False

        if args.position_trainflag:
            x_position, y_position, xita = train_func(args)

            in_name = args.array_position_mat_filename
            out_name = args.save_dir + '/position_out.mat'

            # PPS.save_new_position_map(in_name, out_name, x_position, y_position, merge_position_threshold)
            PPS.save_new_position_map_cubic(in_name, out_name, x_position, y_position, xita, merge_position_threshold)

            args.array_position_mat_filename = out_name
        else:
            _, _, _ = train_func(args)

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_flag
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
       try:
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu,True)
       except RuntimeError as e:
         print(e)

    train_all(args)

if __name__ == '__main__':
    main()