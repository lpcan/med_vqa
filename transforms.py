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

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                     std=[0.229, 0.224, 0.225]),
                                transforms.RandomRotation(10), 
                                transforms.RandomAdjustSharpness(2, p=0.3),
                                transforms.ColorJitter(brightness=0.5, contrast=0.5),
                                transforms.RandomApply(transforms=[AddGaussianNoise(0., 1.)], p=0.3),
                                transforms.Grayscale(num_output_channels=3)])