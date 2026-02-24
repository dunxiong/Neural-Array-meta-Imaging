import torch
from PIL import Image
import numpy as np
import imageio
import os
import skimage.io
from torchvision import transforms
import matplotlib.image as mpimg
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load(image_width_padded, augment=False):
    
    def load_fn(image_file):

        image = imageio.imread(image_file, 'PNG-FI')  # Read image using imageio
        image = image.astype(np.float32) / 65535.  # Normalize the 16-bit image

        meas = image.transpose(2, 0, 1)

        return meas  # Add batch dimension for PyTorch

    return load_fn

class ImageDataset(Dataset):
    def __init__(self, image_dir, image_width_padded, augment=False):
        self.image_dir = image_dir
        self.image_files = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')]
        self.load_fn = load(image_width_padded, augment)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        return self.load_fn(image_file)


def train_dataset_sim(image_width_padded, args):
    dataset = ImageDataset(args.train_dir, image_width_padded, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return loader

def test_dataset_sim(image_width_padded, args):
    dataset = ImageDataset(args.test_dir, image_width_padded, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    return loader
