import torch
from torch import nn, optim

from model import VQAModel
import params
import data_prep

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    print("Starting training...")
    if params.train_val_split == 1:
        data = data_prep.VQADataset(data_dir=params.data)
        trainloader = torch.utils.data.DataLoader(data, batch_size=params.batch_size, shuffle=True)
    else:
        print("Loading data...")
        data = data_prep.VQADataset(data_dir=params.data)
        data_len = len(data)
        train_len = int((params.train_val_split) * data_len)
        test_len = data_len - train_len
        train_set, test_set = torch.utils.data.random_split(data, [train_len, test_len])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)

    # Initialise model, loss function, and optimiser
    print("Initialising model...")
    wordvec_dict = data_prep.create_dict(params.wv_path)
    wordvec_weights = torch.FloatTensor(wordvec_dict)
    print("Finished loading dictionary")
    model = VQAModel(img_feat_size=1024, wordvec_weights=wordvec_weights, q_feat_size=1024)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    # # Start training
    # for epoch in range(params.epochs):
    #     total_loss = 0
    #     total_correct = 0

    #     for batch in trainloader:
    #         v, q, a = batch




    return

if __name__ == '__main__':
    train()