'''
@File      :   AdditiveAttention.py
@Time      :   2024/04/23 22:16:28
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   additive_attention.py completed.
'''

import torch
import torch.nn as nn

def initialize_params(m:nn.Module)->None:
    if isinstance(m,nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class AdditiveAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias = False),
            nn.Softmax(dim = -2)
        )
        self.attention.apply(initialize_params)

    def forward(self, inputs):
        '''
        the input shape as [batchsize, seqlen, d_model]
        after the first Linear, it becomes [batchsize, seqlen, hidden_dim]
        Tanh is an activative function, won't change the shape.
        the second Linear will change it to [batchsize, seqlen, 1]
        the softmax will keep the shape.
        but change the second dim to be a score of attention.
        the input * attention_weight will be a pointwise product. 
        '''
        attention_weight = self.attention(inputs)
        return inputs * attention_weight