import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
from transformers import PreTrainedModel, AutoConfig, AutoModel

class ImgEncoder(nn.Module):
    # VGG-16
    def __init__(self, out_size):
        super(ImgEncoder, self).__init__()
        self.model = models.vgg16(pretrained=True).features # load model and pull out the part we want
        self.linear = nn.Linear(512, out_size) # Reshape to match q_feat size

        #in_feat = self.model.classifier[6].in_features
        #self.model.classifier[6] = nn.Linear(in_feat, out_size) # replace the output layer to give the size we want

    def forward(self, input):
        out = self.model(input) #  shape [batch_size, 512, 8, 8]
        out = out.permute(0, 2, 3, 1) # reshape to [batch_size, 8, 8, 512]
        out = out.view(-1, 64, 512) # reshape to [batch_size, 64, 512]
        #out = torch.tanh(self.linear(out)) # apply linear layer
        out = self.linear(out) # apply linear layer

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

class SAN(nn.Module):
    # Stacked Attention Network
    def __init__(self, num_layers, feat_size, hidden_size):
        super(SAN, self).__init__()

        # Create lists of layers needed for the attention network
        list_img_layer = []
        list_q_layer = []
        list_attn = []
        for i in range(num_layers):
            list_img_layer.append(nn.Linear(feat_size, hidden_size))
            list_q_layer.append(nn.Linear(feat_size, hidden_size))
            list_attn.append(nn.Linear(hidden_size, 1))
        
        # Convert the lists into nn.ModuleLists so that Pytorch can see them
        self.img_layer = nn.ModuleList(list_img_layer)
        self.q_layer = nn.ModuleList(list_q_layer)
        self.attn = nn.ModuleList(list_attn)

        self.num_layers = num_layers
        
    def forward(self, img_vector, q_vector):
        refined_query = q_vector
        
        # Repeat for the desired number of layers
        for i in range(self.num_layers):
            # Pass image and question vectors through layers
            hidden_vec = torch.tanh(self.img_layer[i](img_vector) + self.q_layer[i](refined_query).unsqueeze(1)) # [batch_size, 64, hidden_size]

            # Get attention distribution
            attn_dist = F.softmax(self.attn[i](hidden_vec), dim=1) # [batch_size, 64, 1]

            # Get the weighted image vector and aggregate
            weighted_vec = torch.sum(attn_dist * img_vector, dim=1) # [batch_size, feat_size]

            # Get the refined query vector
            refined_query = weighted_vec + refined_query # [batch_size, feat_size]

        return refined_query
        #return weighted_vec

class AnsGenerator(nn.Module):
    # Classification method
    def __init__(self, embed_size, pos_ans, drop=0):
        super(AnsGenerator, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop)
        self.lin1 = nn.Linear(embed_size,  pos_ans)
        #self.gap = nn.AdaptiveAvgPool2d(output_size=1) # Global Average Pooling
        self.lin2 = nn.Linear(pos_ans, pos_ans)
    def forward(self, input):
        x = self.dropout(input)
        x = self.lin1(x)
        #x = self.gap(x)
        x = self.relu(x)
        x = self.lin2(x)
        return x

class VQAModel(nn.Module):
    # ImgEncoder & QEncoder -> Feature fusion -> AnsGenerator

    def __init__(self, feat_size, out_size, num_layers=3, dropout=0):
        super(VQAModel, self).__init__()
        self.img_encoder = ImgEncoder(feat_size)
        self.q_encoder = QEncoder(feat_size)
        self.attention = SAN(num_layers, feat_size, 256)
        self.classifier = AnsGenerator(feat_size, out_size, drop=dropout)

    def forward(self, v, q):
        img_feat = self.img_encoder(v) # [batch_size, 64, feat_size]
        q_feat = self.q_encoder(q) # [batch_size, feat_size]
        fused_feat = self.attention(img_feat, q_feat) # [batch_size, feat_size]

        # Find the most likely answer
        ans = self.classifier(fused_feat)
        
        return ans

        
