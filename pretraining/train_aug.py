# Self-supervised pretraining of VGG-16
import numpy as np
import torch
import torch.nn as nn
import sys
import os
from torchvision import models
from pytorch_metric_learning.losses import NTXentLoss

from preprocess import UnlabelledImages, transform

batch_size = 128
epochs = 10
learning_rate = 1e-4 # 0.05 * batchsize/256?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImgEncoder(nn.Module):
    # VGG-16
    def __init__(self, out_size):
        super(ImgEncoder, self).__init__()
        self.model = models.vgg16(pretrained=True).features # load model and pull out the part we want
        # Load the pretrained model
        #self.model = models.vgg16()
        #self.model.load_state_dict(torch.load("pretraining/ss_vgg16.pth"))
        #self.model = self.model.features # Pull out just the part we want (for SAN)
        self.linear = nn.Linear(512, out_size) # Reshape to match q_feat size

        #in_feat = self.model.classifier[6].in_features
        #self.model.classifier[6] = nn.Linear(in_feat, out_size) # replace the output layer to give the size we want

    def forward(self, input):
        out = self.model(input) #  shape [batch_size, 512, 8, 8]
        out = out.permute(0, 2, 3, 1) # reshape to [batch_size, 8, 8, 512]
        out = out.view(-1, 64, 512) # reshape to [batch_size, 64, 512]
        out = self.linear(out) # apply linear layer

        return out

def train():

    # Load the data - in three separate folders but we want to just use all of it
    data_dir = "/srv/scratch/z5214005/roco-dataset/data/"
    train_data = UnlabelledImages(data_dir+"train/radiology/images/*", transform)
    val_data = UnlabelledImages(data_dir+"validation/radiology/images/*", transform)
    test_data = UnlabelledImages(data_dir+"test/radiology/images/*", transform)

    dataset = torch.utils.data.ConcatDataset([train_data, val_data, test_data])
    #dataset = torch.utils.data.Subset(test_data, np.arange(0, 1000)) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    #model = models.vgg16(pretrained=True) # Start with a pretrained (on ImageNet) VGG-16 model
    model = ImgEncoder(256)

    if os.path.isfile("ss_vgg16.pth"):
        model.load_state_dict(torch.load("ss_vgg16.pth"))
        #model = torch.load("ss_vgg16")
    model.to(device)
    loss_func = NTXentLoss(temperature=0.5)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('----------')
        
        total_loss = 0
        
        for i, batch in enumerate(dataloader):
            # Prepare the data
            t1, t2 = batch # Each sample is transformed twice
            data = torch.cat((t1, t2), dim=0) # Turn into a batch of data of length 2*batch_size
            data = data.to(device)

            # Pass through the model
            embeddings = model(data) # [batch_size * 2, 64, 512]
            embeddings = embeddings.view(len(t1) * 2, -1) # [batch_size * 2, 64*512]
            
            # Label positive pairs
            labels = torch.from_numpy(np.arange(len(t1)))
            pair_labels = torch.cat((labels, labels))
            
            # Calculate the loss
            loss = loss_func(embeddings, pair_labels)
            
            # Backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
        
        print(f"Loss = {total_loss}")

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), "ss_vgg16.pth")

    # Save the model
    torch.save(model.state_dict(), "ss_vgg16.pth")

if __name__ == '__main__':
    train()