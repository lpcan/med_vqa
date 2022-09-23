# Self-supervised pretraining of VGG-16
import numpy as np
import torch
import sys
from torchvision import models
from pytorch_metric_learning.losses import NTXentLoss

from preprocess import UnlabelledImages, transform

batch_size = 128
epochs = 3
learning_rate = 0.05 # 0.05 * batchsize/256?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():

    # Load the data - in three separate folders but we want to just use all of it
    data_dir = "/srv/scratch/z5214005/roco-dataset/data/"
    #train_data = UnlabelledImages(data_dir+"train/radiology/images/*", transform)
    #val_data = UnlabelledImages(data_dir+"validation/radiology/images/*", transform)
    test_data = UnlabelledImages(data_dir+"test/radiology/images/*", transform)

    #dataset = torch.utils.data.ConcatDataset([train_data, val_data, test_data])
    dataset = torch.utils.data.Subset(test_data, np.arange(0, 128)) 
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    model = models.vgg16(pretrained=True) # Start with a pretrained (on ImageNet) VGG-16 model
    model.to(device)
    loss_func = NTXentLoss()
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
            embeddings = model(data)

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
            torch.save(model, "ss_vgg16")

    # Save the model
    torch.save(model, "ss_vgg16")

if __name__ == '__main__':
    train()