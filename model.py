import torch
import torch.nn as nn
from torchvision import models
from transformers import PreTrainedModel, AutoConfig, AutoModel

import params

class ImgEncoder(nn.Module):
    # VGG-16
    def __init__(self, out_size):
        super(ImgEncoder, self).__init__()
        self.model = models.vgg16(pretrained=True) # load model

        in_feat = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_feat, out_size) # replace the output layer to give the size we want

        # Load the pretrained state_dict
        pretrained_dict = torch.load("pretraining/ss_vgg16.pth")
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def forward(self, input):
        out = self.model(input) # we are finetuning the entire model, so no need to freeze parameters
        # normalisation???
        return out

class QEncoder(PreTrainedModel):
    # BERT Transformer
    def __init__(self, q_feat_size): 
        config = AutoConfig.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        super(QEncoder, self).__init__(config)
        # Instantiate the BERT model
        self.model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
        self.linear = nn.Linear(768, q_feat_size)
        
    def forward(self, input):
        output = self.model(input) # dictionary of last_hidden_state: [batch_size, q_length, 768], pooler_output: [batch_size, 768] 
        output = output[1] # [batch_size, 768 (hidden_size)]
        output = self.linear(output) # [batch_size, q_feat_size]
        return output

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

    def forward(self, v, q = None):
        if q is None:
            # v and q are together and need to be reshaped. Original shape is [batch_size, 3*256*256 + 16]
            input = v
            batch_size = input.shape[0]
            v = input[:, :(3*256*256)].view((batch_size, 3, 256, 256))
            q = input[:, (3*256*256):].view((batch_size, -1)).int()

        img_feat = self.img_encoder(v) # [batch_size, img_feat_size]
        q_feat = self.q_encoder(q) # [batch_size, q_feat_size]
        fused_feat = torch.cat((img_feat, q_feat), dim=1) # [batch_size, img_feat_size + q_feat_size]

        # Classification method
        ans = self.classifier(fused_feat)
        
        return ans