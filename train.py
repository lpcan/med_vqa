import torch
from torch import nn, optim

from model import VQAModel
import params
import data_prep

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    if params.train_val_split == 1:
        data = data_prep.VQADataset(data_dir=params.data)
        trainloader = torch.utils.data.DataLoader(data, batch_size=params.batch_size, shuffle=True)
    else:
        print("Loading data...")
        data = data_prep.VQADataset(data_dir=params.data, transform=params.transform)
        data_len = len(data)
        train_len = int((params.train_val_split) * data_len)
        test_len = data_len - train_len
        train_set, test_set = torch.utils.data.random_split(data, [train_len, test_len])
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=False)
        testloader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)

    # Initialise model, loss function, and optimiser
    print("Initialising model...")
    wordvec_dict = data_prep.create_dict(params.wv_path)
    wordvec_weights = torch.FloatTensor(wordvec_dict.vectors)
    model = VQAModel(img_feat_size=1024, wordvec_weights=wordvec_weights, q_feat_size=1024)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    # Start training
    for epoch in range(params.epochs):
        print(f'Epoch {epoch+1}/{params.epochs}')
        print('----------')
        total_loss = 0
        total_correct = 0

        for batch in trainloader:
            v, q, a = batch
            v = v.to(device)
            q = q.to(device)
            a = a.to(device)

            # Get the predictions for this batch
            pred = model(v, q)

            # Calculate the loss
            loss = criterion(pred, a)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward() # compute gradients
            optimiser.step() # update weights

            # Get the output answer
            #_, answer = torch.max(pred, 1)

            total_loss += loss.item()
            # total_correct += # how to calculate "total correct?"

        model_accuracy = 0.0 # implement some accuracy metric here

        print(f"Total_correct: {total_correct}\nLoss: {total_loss:.2f}\nAcc: {model_accuracy:.2f}")

        if epoch % 1 == 0: # validate after every epoch?
            if params.train_val_split < 1:
                test_network(model,testloader)

    if params.train_val_split < 1:
        test_network(model,testloader)
    #torch.save(net.state_dict(), 'savedModel.pth')
    #print("   Model saved to savedModel.pth")

def test_network(model, testloader):
    model.eval()
    total_correct = 0

    # Write the output somewhere?
    f = open('valid_log.txt', 'a+')
    f.write('--------------')

    with torch.no_grad():
        for data in testloader:
            v, q, a = data
            v = v.to(device)
            q = q.to(device)
            a = a.to(device)

            pred = model(v, q)
            _, output = torch.max(pred, 1)
            f.write(q + '|' + output + '|' + a)
            # something with total correct here

    model_accuracy = 0 # something
    print(f"Accuracy on #num# of images: {model_accuracy:.2f}")  
    f.close()
    model.train()

if __name__ == '__main__':
    train()