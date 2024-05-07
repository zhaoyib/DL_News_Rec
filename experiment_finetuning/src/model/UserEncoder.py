'''
@File      :   UserEncoder.py
@Time      :   2024/04/23 22:16:59
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   create userEncoder.py
'''


import torch
import torch.nn as nn
from utils.logger import logger_wrapper
from model.AdditiveAttention import AdditiveAttention

class UserEncoder(nn.Module):
    def __init__(self, d_model, num_heads = 8, hidden_dim = 256 ) -> None:
        super().__init__()
        self.d_model = d_model
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim = self.d_model, num_heads = num_heads,
            batch_first = True
        )
        self.additive_attention = AdditiveAttention(d_model, hidden_dim)
        self.logger = logger_wrapper("User Encoder")
        
    def forward(self, histories_text, histories_mask, histories_imgs, news_encoder):
        '''
        using the histories part in batch to extract the user's feature.
        using the news_encoder(which is the MultiModalEncoder in fact)
        news_histories is a tensor with the shape [batchsize, seqlen, emb_dim]
        multihead attention will keep the shape
        additive attention will also keep the shape
        but the meaning has changed.
        then, using the sum to catch the user interest.
        ''' 
        batch_size, histories_len, seq_len = histories_text.shape
        _, _, channel, height, weight = histories_imgs.shape
        histories_text = histories_text.view(batch_size * histories_len, seq_len)
        histories_mask = histories_mask.view(batch_size * histories_len, seq_len)
        histories_imgs = histories_imgs.view(batch_size * histories_len, channel, height,weight)
        
        # input of news_encoder is (bs*his_len, seqlen)
        news_histories = news_encoder(histories_text, histories_mask, histories_imgs)
    
        # input of attention is (bs*his_len, hidden_dim)
        multihead_attention_output, _ = self.multihead_attention(
            news_histories, news_histories, news_histories
        )
        additive_attention_output = self.additive_attention(
            multihead_attention_output
        )
        additive_attention_output = additive_attention_output.view(batch_size, histories_len,-1)
        
        output = torch.sum(additive_attention_output, dim = 1)
        # output with batchsize, output_dim, batchsize means the num of users.
        return output