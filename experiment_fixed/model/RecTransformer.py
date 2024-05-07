'''
@File      :   RecTransformer.py
@Time      :   2024/04/13 21:44:03
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   the encoder only model
'''



import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk") 

class Positional_Embedding(nn.Module):
    '''
    as the report, it uses the formula to calculate the Positional_Embedding
    '''
    def __init__(self, d_model:int, dropout:float, max_len=5000):
        super(Positional_Embedding,self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        positional_embedding = torch.zeros(max_len,d_model)#initialize the tensor
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        positional_embedding[:,0::2] = torch.sin(position * div_term)#even index
        positional_embedding[:,1::2] = torch.cos(position * div_term)#odd index
        positional_embedding = positional_embedding.unsqueeze(0)
        #[max_len,d_model] to [1,max_len,d_model],for batch size
        self.register_buffer('pe',positional_embedding)

    def forward(self,x):
        '''
        x is the vector after embedding. it is [seq_length,d_model]
        if in batch.it is [batch_size,seq_length,d_model]
        '''
        #print(x.device)
        x = x + Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)

#Part 3: multi-head attention
def attention(query, key, value, mask = None, dropout = None):
    '''
    query,key,value are the tensors. Q and K has the same size
    V: [batch_size, num_head, seq_length_1, d_K] = K.size
    Q: [batch_size, num_head, seq_length_2, d_K]
    d_K is a new hyperparameter. to set the dimension of words.
    it is a little different from my report.
    '''
    d_k = query.size(-1)
    #print("qkv:",query.shape,key.shape)
    scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
    #print("score:",scores.shape)
    #print("mask:",mask.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        #after softmax, masked place will set as 0
    p_attention = F.softmax(scores,dim=-1)
    #[batch_size, num_head, seq_length_2, seq_length_1]
    if dropout is not None:
        p_attention = dropout(p_attention)
    return torch.matmul(p_attention,value),p_attention

#before class 3,we need a clones function
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#class 3: multi-head attention
class Multi_Head_Attention(nn.Module):
    def __init__(self, num_head, d_model, dropout = 0.1):
        super(Multi_Head_Attention,self).__init__()
        assert d_model % num_head == 0 # why we got this assert? without it, what will change?
        self.d_K = d_model // num_head 
        self.num_head = num_head
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attention = None
        self.dropout = nn.Dropout(p = dropout)
    def forward(self,query,key,value,mask = None):
        '''
        query:[batch_size,seq_len_2,d_model]
        key:[batch_size,seq_len_1,d_model]  = value
        '''
        #print('multi head attention forward')
        batch_size = query.size(0)
        query,key,value = [l(x).view(batch_size,-1,self.num_head,self.d_K)
                           .transpose(1,2) for l,x in 
                           zip(self.linears,(query,key,value))]
        x,self.attention = attention(query,key,value,mask=mask,dropout=self.dropout)
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.num_head*self.d_K)
        return self.linears[-1](x)

#class 4: sublayerconnection
#before that we need the class 7: Layer_Norm
class Layer_Norm(nn.Module):
    def __init__(self,features,eps=1e-6):
        super(Layer_Norm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        #two parameters. vector. trainable. [features=d_model]
        self.eps = eps
    def forward(self,x):
        #print("layer norm forward")
        #x.size = [batch_size, seq_length, d_model]
        mean = x.mean(-1,keepdim = True)
        std = x.std(-1,keepdim = True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2

#class 4: sublayerconnection
class Sub_Layer_Connection(nn.Module):
    '''
    residual connection and a layer norm.
    '''
    def __init__(self,size,dropout):
        super(Sub_Layer_Connection,self).__init__()
        self.norm = Layer_Norm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

#class 5: Positionwise_Feed_Forward
class Positionwise_Feed_Forward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(Positionwise_Feed_Forward,self).__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        #first fcc, [d_model,d_ff]
        self.w_2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        #print("pff forward")
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

#class 6: Encoder_Layer
class Encoder_Layer(nn.Module):
    def __init__(self,size:int,self_attention,feed_forward,dropout):
        '''
        size is d_model
        self_attention is the object Multi_Head_Attention, is the first sublayer
        feed_forward is the object Postionwise_Feed_Forward, seconde sublayer
        '''
        super(Encoder_Layer,self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(Sub_Layer_Connection(size,dropout),2)
        self.size = size
    def forward(self,x,mask):
        x = self.sublayer[0](x,lambda x:self.self_attention(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

#class 7 is defined above
#class 8:Encoder
class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder,self).__init__()
        self.layers = clones(layer,N)
        self.norm = Layer_Norm(layer.size)
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    #it won't change the size of x

class RecTransformer(nn.Module):
    def __init__(self, encoder_layer_num = 6, d_model = 768, num_heads = 8,
                dff = 2048, dropout = 0.1, batch_first = True,device="cuda:0"):
        super(RecTransformer,self).__init__()
        self.num_head = num_heads
        self.d_model = d_model
        self.d_ff = dff
        self.dropout = dropout
        self.N = encoder_layer_num
        c = copy.deepcopy#deep copy/clone
        attention  = Multi_Head_Attention(self.num_head,self.d_model)#object
        ff = Positionwise_Feed_Forward(self.d_model,self.d_ff,self.dropout)#object
        position = Positional_Embedding(self.d_model,self.dropout)
        self.encoder = Encoder(Encoder_Layer(self.d_model,c(attention),c(ff),self.dropout),self.N)
        self.source_embedder = c(position)
        self.device = device
        self.output_layer = nn.Sequential(
            nn.Linear(d_model,256),
            nn.Linear(256,16),
            nn.Linear(16,2)
        )
        
    def forward(self, input_batch):
        batch = input_batch["batch"]
        mask = input_batch["mask"]
        tgt = input_batch["target"]
        mask = mask.to(self.device)
        vec = self.encoder(self.source_embedder(batch),mask)
        tgt = F.one_hot(tgt,2).to(self.device).float()
        to_pred = torch.mean(vec,dim=1) # kill the sequence dimension.
        out_put = self.output_layer(to_pred) # batchsize, 1, 768 to batchsize, 1, 2
        out_put = out_put.squeeze(dim=1) # kill the sequence dimension.
        weight = torch.tensor([1,25]).to(self.device)
        loss = nn.CrossEntropyLoss(weight=weight)
        #print(out_put.dtype)
        #print(tgt.dtype)
        #print(out_put.shape)
        #print(tgt.shape)
        res_dict = {"result":out_put, "loss":loss(out_put,tgt)}
        return res_dict

    def model_test(self, input_batch):
        """
        use it for test or inference.
        """
        with torch.no_grad():
            batch = input_batch["batch"]
            mask = input_batch["mask"].to(self.device)
            vec = self.encoder(self.source_embedder(batch),mask=None)
            to_pred = torch.mean(vec,dim=1)
            output = self.output_layer(to_pred)
            output = output.squeeze(dim=1)
            return output

def build_model(config):
    model_set = config.model_set
    encoder_layer_num = model_set.get("encoder_num",6)
    d_model = model_set.get("d_model",768)
    num_heads = model_set.get("n_heads",8)
    dff = model_set.get("dim_ffm",2048)
    dropout = model_set.get("dropout",0.1)
    batch_first = model_set.get("batch_first",True)
    device = config.device[0]
    model = RecTransformer(encoder_layer_num,d_model,num_heads,dff,dropout,batch_first,device)
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model

# def _load_checkpoint(config):
#     model_file = config.ckp_path + f"_{config.start_epoch}" + ".pth" # ckp_path is a checkpoint_date_epoch.pth format.
#     check_pnt = torch.load(model_file, map_location= None)#map_location is none
#     # load parameter of model
#     self.model.load_state_dict(check_pnt["model"])
#     # load parameter of optimizer
#     self.optimizer.load_state_dict(check_pnt["optimizer"])
#     # other parameter
#     self.start_epoch = check_pnt["start_epoch"]
#     self.best_valid_cor_f1 = check_pnt["best_valid_cor_f1"]
#     self.best_test_cor_f1 = check_pnt["best_test_cor_f1"]