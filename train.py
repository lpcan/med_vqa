import torch
from torch import nn, optim

from model import VQAModel
import params
import data_prep

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    if params.train_val_split == 1:
        data = data_prep.VQADataset(data_dir=params.data, transform=params.transform)
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
    wordvec_dict = data_prep.create_dict(params.wv_path, params.data)
    wordvec_weights = torch.FloatTensor(wordvec_dict.vectors)
    num_answers = len(data.unique_answers)
    model = VQAModel(img_feat_size=512, wordvec_weights=wordvec_weights, q_feat_size=512, out_size = num_answers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-5)

    # Start training
    for epoch in range(params.epochs):
        print(f'Epoch {epoch+1}/{params.epochs}')
        print('----------')
        total_loss = 0
        total_correct = 0
        total_samples = 0
        print(f"{len(trainloader)} batches")
        for i, batch in enumerate(trainloader):
            print(f"loading batch {i}")
            v, q, a = batch
            v = v.to(device)
            q = q.to(device)
            a = a.to(device)

            # Get the predictions for this batch
            pred = model(v, q)
            
            # Get the output answer
            answer = torch.argmax(pred, dim=1)

            # Calculate the loss
            loss = criterion(pred, a)

            # Backpropagation
            optimiser.zero_grad()
            loss.backward() # compute gradients
            optimiser.step() # update weights

            total_loss += loss.item()
            print(f"{loss.item()}")
            total_correct += answer.eq(a).sum().item()
            total_samples += a.size(0)

        model_accuracy = total_correct / total_samples # implement some accuracy metric here

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
    total_samples = 0

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
            answer = torch.argmax(pred, dim=1)

            answers = list(testloader.dataset.dataset.unique_answers)
            output = [answers[label] for label in answer]
            a_word = [answers[label] for label in a]

            for i in range(len(q)):
                vocab = testloader.dataset.dataset.question_vocab
                f.write(vocab.idx_to_sentence(q[i]) + '|' + output[i] + '|' + a_word[i] + '\n')
            total_samples += a.size(0)
            total_correct += (answer == a).sum().item()


    model_accuracy = total_correct/total_samples
    print(f"Accuracy on #num# of images: {model_accuracy:.2f}")  
    f.close()
    model.train()

if __name__ == '__main__':
    train()