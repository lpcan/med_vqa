import torch
import torch.nn as nn
from torchvision import models
from transformers import PreTrainedModel, AutoConfig, AutoModel

class ImgEncoder(nn.Module):
    # VGG-16
    def __init__(self, out_size):
        super(ImgEncoder, self).__init__()
        self.model = models.vgg16(pretrained=True) # load model
        in_feat = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_feat, out_size) # replace the output layer to give the size we want

    def forward(self, input):
        out = self.model(input) # we are finetuning the entire model, so no need to freeze parameters
        # normalisation???
        return out

class QEncoder(PreTrainedModel):
    # BioBERT Transformer
    def __init__(self, q_feat_size): 
        config = AutoConfig.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        super(QEncoder, self).__init__(config)
        # Instantiate the BioBERT model
        self.model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        self.linear = nn.Linear(768, q_feat_size)
        
    def forward(self, input):
        output = self.model(input) # dictionary of last_hidden_state: [batch_size, q_length, 768], pooler_output: [batch_size, 768] 
        output = output[1] # [batch_size, 768 (hidden_size)]
        output = self.linear(output) # [batch_size, q_feat_size]
        return output

"""
class AnsGenerator(nn.Module):
    # LSTM decoder. Not really sure about this
    def __init__(self, wordvec_weights):
        super(AnsGenerator, self).__init__()
        self.hidden_size = wordvec_weights.shape[1]
        self.out_size = wordvec_weights.shape[0]

        self.embedding = nn.Embedding.from_pretrained(wordvec_weights)
        self.embedding.requires_grad = False # don't train the embedding
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.out_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden): # hidden state is the final state of the encoder
        word_vec = self.embedding(input)
        output, hidden = self.lstm(word_vec, hidden)
        output = self.softmax(self.out(output[0])) # Why output[0]?
        return output, hidden
"""  
class AnsGenerator(nn.Module):
    # Classification method
    def __init__(self, embed_size, pos_ans, drop=0):
        super(AnsGenerator, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.lin1 = nn.Linear(embed_size,  pos_ans)
        self.lin2 = nn.Linear(pos_ans, pos_ans)
    def forward(self, input):
        x = self.dropout(input)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

class VQAModel(nn.Module):
    # ImgEncoder & QEncoder -> Feature fusion -> AnsGenerator

    def __init__(self, img_feat_size, q_feat_size, out_size, dropout=0):
        super(VQAModel, self).__init__()
        self.img_encoder = ImgEncoder(img_feat_size)
        self.q_encoder = QEncoder(q_feat_size)
        self.classifier = AnsGenerator(img_feat_size + q_feat_size, out_size, drop=dropout)

    def forward(self, v, q):
        img_feat = self.img_encoder(v) # [batch_size, img_feat_size]
        q_feat = self.q_encoder(q) # [batch_size, q_feat_size]
        fused_feat = torch.cat((img_feat, q_feat), dim=1) # [batch_size, img_feat_size + q_feat_size]
        
        # this is not right
        # ans = self.decoder(fused_feat, hidden_state)

        # Classification method
        ans = self.classifier(fused_feat)
        
        return ans

        
