import torch
from torch import nn, optim
import matplotlib.pyplot as plt

from model import VQAModel
import params
import data_prep
import vocab_helper

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train():
    print("Loading data...")

    if params.k_folds > 0:
        # Split the data to perform k fold cross validation
        k = params.k_folds
        vocab = vocab_helper.Vocab([params.train_data])
        ans_translator = vocab_helper.Ans_Translator([params.train_data])

        data = data_prep.VQADataset(data_dir=params.train_data, img_dir=params.train_img_dir, vocab=vocab, ans_translator=ans_translator)
        split_len = len(data) // k 
        splits = ([split_len] * (k-1)) + [len(data) - (split_len * (k-1))] # Lengths of each dataset
        sets = torch.utils.data.random_split(data, splits, generator=torch.Generator().manual_seed(42)) # Split the data randomly, seeded for reproducibility
    else:
        k = 1 # So that it still works without setting k_fold cross validation

    fold_results = []

    # Perform the training
    for fold in range(k):
        if k > 1:
            print(f"#################### FOLD {fold} ####################")
            # Prepare the data for this fold
            test_data = data_prep.DatasetFromSubset(sets[fold], transform=params.val_transform)
            train_datasets = []
            for i in range(k):
                if i == fold: continue
                train_datasets.append(data_prep.DatasetFromSubset(sets[i], transform=params.train_transform))
            train_data = torch.utils.data.ConcatDataset(train_datasets) # Concatenate the training datasets

            trainloader = torch.utils.data.DataLoader(train_data, batch_size=params.batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=params.batch_size, shuffle=False)

        elif params.train_data != params.val_data: 
            # Separate training and validation sets provided
            vocab = vocab_helper.Vocab([params.train_data, params.val_data])
            ans_translator = vocab_helper.Ans_Translator([params.train_data, params.val_data])
            train_data = data_prep.VQADataset(data_dir=params.train_data, img_dir=params.train_img_dir, vocab=vocab, ans_translator=ans_translator, transform=params.train_transform)
            test_data = data_prep.VQADataset(data_dir=params.val_data, img_dir=params.val_img_dir, vocab=vocab, ans_translator=ans_translator, transform=params.val_transform)
            trainloader = torch.utils.data.DataLoader(train_data, batch_size=params.batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(test_data, batch_size=params.batch_size, shuffle=False)
        else:
            # Split the data according to train_val_split
            vocab = vocab_helper.Vocab([params.train_data])
            ans_translator = vocab_helper.Ans_Translator([params.train_data])
            data = data_prep.VQADataset(data_dir=params.train_data, img_dir=params.train_img_dir, vocab=vocab, ans_translator=ans_translator, transform=params.train_transform)
            data_len = len(data)
            train_len = int((params.train_val_split) * data_len)
            test_len = data_len - train_len
            train_subset, test_subset = torch.utils.data.random_split(data, [train_len, test_len])
            train_set = data_prep.DatasetFromSubset(train_subset, transform = transforms.train_transform)
            test_set = data_prep.DatasetFromSubset(test_subset, transform = transforms.val_transform)

            trainloader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
            testloader = torch.utils.data.DataLoader(test_set, batch_size=params.batch_size, shuffle=False)

        # Initialise model, loss function, and optimiser
        print("Initialising model...")
        num_answers = len(ans_translator.answer_list)
        model = VQAModel(img_feat_size=256, q_feat_size=256, out_size = num_answers, dropout=0.5)
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
            #print(f"{len(trainloader)} batches")
            for i, batch in enumerate(trainloader):
                #print(f"loading batch {i}")
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
                #print(f"{loss.item()}")
                total_correct += answer.eq(a).sum().item()
                total_samples += a.size(0)

            model_accuracy = total_correct / total_samples # implement some accuracy metric here

            print(f"Total_correct: {total_correct}\nLoss: {total_loss:.2f}\nAcc: {model_accuracy:.2f}")

            if epoch % 10 == 0: # validate after every 10 epochs
                if params.train_val_split < 1:
                    test_network(model, testloader, vocab, ans_translator)

        if params.train_val_split < 1 or k > 1:
            acc = test_network(model, testloader, vocab, ans_translator)
        
        if k > 1:
            fold_results.append(acc)

        #torch.save(net.state_dict(), 'savedModel.pth')
        #print("   Model saved to savedModel.pth")
    
    # Print the model's final results
    if k > 1:
        print("#################### FINAL RESULTS ####################")
    for i in range(len(fold_results)):
        print(f"Fold {i} Acc: {fold_results[i]}")
        
def test_network(model, testloader, vocab, ans_translator):
    model.eval()
    total_correct = 0
    total_samples = 0

    # Write the output somewhere?
    f = open('valid_log.txt', 'w+')
    f.write('--------------')

    with torch.no_grad():
        for data in testloader:
            v, q, a = data
            v = v.to(device)
            q = q.to(device)
            a = a.to(device)

            pred = model(v, q)
            answer = torch.argmax(pred, dim=1)

            output = [ans_translator.label_to_ans(label) for label in answer]
            gt = [ans_translator.label_to_ans(label) for label in a]

            for i in range(len(q)):
                f.write(vocab.idx_to_sentence(q[i]) + '|' + output[i] + '|' + gt[i] + '\n')
            total_samples += a.size(0)
            total_correct += (answer == a).sum().item()

    model_accuracy = total_correct/total_samples
    print(f"Accuracy on {total_samples} images: {model_accuracy:.2f}")  
    f.close()
    model.train()

    return model_accuracy

if __name__ == '__main__':
    train()
