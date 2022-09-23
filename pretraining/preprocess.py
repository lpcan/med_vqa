import glob
import sys
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

import matplotlib.pyplot as plt

##### TRANSFORMS #####
# Create a custom transform to add a small amount of noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ +  '(mean={0}, std={1}'.format(self.mean, self.std)

transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.263, 0.262, 0.262], 
                                    std=[0.262, 0.262, 0.262]),
            transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9,1.1))], p=0.4),
            transforms.RandomAffine(20, translate=(0.05, 0.05), shear=(0.2, 0.2, 0.2, 0.2)), 
            transforms.RandomApply(transforms=[transforms.GaussianBlur(5)], p=0.5),
            transforms.RandomApply(transforms=[AddGaussianNoise(0., 0.1)], p=0.4),
            transforms.Grayscale(num_output_channels=3)])

##### DATASET CLASS #####

class UnlabelledImages(data.Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.files = glob.glob(data_dir)
        self.transform = transform

    def __getitem__(self, idx):
        try:
            # Open the image
            img = Image.open(self.files[idx])

            # Resize to 256x256
            img = img.resize((256,256))
        except OSError as err:
            print(err)
            print(f"Error with file {self.files[idx]}")
            sys.exit(1)

        # Make an array
        img = np.asarray(img)

        # Check that we have the right number of channels
        if len(img.shape) == 2:
            # Need to make it a 3 channeled image
            img = np.stack((img,)*3, axis=-1)

        # Transform twice
        t1 = self.transform(img)
        t2 = self.transform(img)

        # Return both images
        return t1, t2
    
    def __len__(self):
        return len(self.files)