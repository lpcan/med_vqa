import torch
import torchvision.transforms as transforms

# Create a custom transform to add a small amount of noise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ +  '(mean={0}, std={1}'.format(self.mean, self.std)

train_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.263, 0.262, 0.262], 
                                                     std=[0.262, 0.262, 0.262]),
                                transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=(0.9, 1.0), contrast=(0.9,1.0))], p=0.4),
                                transforms.RandomRotation(10), 
                                transforms.RandomApply(transforms=[AddGaussianNoise(0., 0.1)], p=0.4),
                                transforms.Grayscale(num_output_channels=3)])

# Separate transform for validation
val_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.263, 0.262, 0.262], 
                                                     std=[0.262, 0.262, 0.262]),
                                    transforms.Grayscale(num_output_channels=3)])