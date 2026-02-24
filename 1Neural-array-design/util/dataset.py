import numpy as np
import tensorflow as tf
import os
import matplotlib.image as mpimg
import scipy.io as sio


def generate_mask(H):

    mask_out = np.ones((512,512))
    M = H//2 -256
    mask_out = np.pad(mask_out,((M,M),(M,M)),'constant',constant_values=(1e-8,1e-8))

    return mask_out

def load(image_width_padded, mask, augment):
    # image_width = Width for image content
    # image_width_padded = Width including padding to accomodate PSF
    def load_fn(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        image = tf.cast(image, tf.float32)
        image = image / 255.
        image = image**(1.8)

        H1, W1= tf.shape(image)[0],tf.shape(image)[1]
        image = tf.image.resize(image, [tf.cast(H1/2,dtype=tf.int32), tf.cast(W1/2,dtype=tf.int32)], antialias=True)
        
        if augment:
            image = tf.image.random_crop(value=image, size=[512, 512, 3])
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

        image = tf.image.resize_with_crop_or_pad(image, image_width_padded, image_width_padded)
        image = image * mask
        
        return image # Input and GT
    
    return load_fn

def train_dataset_sim(image_width_padded, args):
    mask = generate_mask(image_width_padded)
    mask = mask[:,:,np.newaxis]
    load_fn = load(image_width_padded, mask, augment=True)
    ds = tf.data.Dataset.list_files(args.train_dir+'*.jpg')
    ds = ds.map(load_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(20)
    ds = ds.repeat() # Repeat forever
    ds = ds.batch(2) # Batch size = 1
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def test_dataset_sim(image_width_padded, args):
    mask = generate_mask(image_width_padded)
    mask = mask[:,:,np.newaxis]
    load_fn = load(image_width_padded, mask ,augment=False)
    ds = tf.data.Dataset.list_files(args.test_dir+'*.jpg', shuffle=False)
    ds = ds.map(load_fn, num_parallel_calls=None)
    ds = ds.batch(2) # Batch size = 1
    return ds


def file_match(s, root):
    dirs = []
    matchs = []
    for current_name in os.listdir(root):
        add_root_name = os.path.join(root, current_name)
        if os.path.isdir(add_root_name):
            dirs.append(add_root_name)
        elif os.path.isfile(add_root_name) and (s == os.path.splitext(add_root_name)[-1][1:]):
            matchs.append(add_root_name)
    for dir in dirs:
        file_match(s, dir)
    return matchs


def dataset_load(target_dir, filetype, color=True):
    print("start data loading.....")
    file_lists = file_match(filetype, target_dir)

    input_image = [None] * 500

    if filetype == 'mat':

        for i in range(len(file_lists)):
            data_temp = sio.loadmat(file_lists[i])
            # file_key = str(print("%04d" % i))
            input_image_temp = data_temp['data'].astype(np.float32)
            input_image[i] = input_image_temp
            print(input_image[i].shape)
    else:
        for i in range(len(file_lists)):
            data_temp = mpimg.imread(file_lists[i])
            if color is False:
                data_temp = np.mean(data_temp, axis=2, keepdims=True)
            input_image[i] = data_temp.astype(np.float32)
            print(input_image[i].shape)

    print(len(input_image))
    image_shape = input_image[0].shape
    num_channels = image_shape[2]

    print("data load finished!")

    return (input_image, len(file_lists), num_channels)


def dataset_preprocess(input_image, patch_size, augment=True):
    image = tf.cast(input_image, tf.float32)  # Shape [height, width, channels]
    image = tf.expand_dims(image, 0)
    image /= 4095.  # 255

    batch, H, W, num_channels = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], tf.shape(image)[3]

    if augment:
        scaled_H = tf.random.uniform([1], minval=patch_size, maxval=H + 1, dtype=tf.int32, seed=None, name=None)
        scaled_W = tf.random.uniform([1], minval=patch_size, maxval=W + 1, dtype=tf.int32, seed=None, name=None)

        image = tf.image.resize(image, [scaled_H[0], scaled_W[0]], antialias=True)

        image = tf.image.random_crop(value=image, size=[batch, patch_size, patch_size, num_channels])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    else:
        image = tf.image.resize_with_crop_or_pad(image, patch_size, patch_size)

    return image




