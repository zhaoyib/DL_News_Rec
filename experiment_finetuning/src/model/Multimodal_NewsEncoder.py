'''
@File      :   Multimodal_NewsEncoder.py
@Time      :   2024/04/24 14:52:02
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   Create the Multimodal_NewsEncoder.py
'''

import torch
from torch import nn
from utils.logger import logger_wrapper
from transformers import AutoConfig, AutoModel
from model.AdditiveAttention import AdditiveAttention

'''
Total work flow:
    1. encode the news text.
    2. encode the pic if pic exists.
    3. hadamard product.
    4. return the embedded vector.

Total project structure:
    1. visual model
    2. text model
    3. NewsEncoder
    4. get_NewsEncoder()
'''

class TextEncoder(nn.Module):
    def __init__(self, pretrained:str = "bert-base-uncased",
                 head_nums: int = 8,
                 additive_hidden_dim = 256):
        '''
        Initial the TextEncoder.
        Configs:
            pretrained: the name of pretrained model from huggingface, or a path.
            head_nums: the num of head for the multihead attention.
            additive_hidden_dim: the hidden dim for additive attention.
        '''
        super().__init__()
        self.pretrained_textual_model = AutoModel.from_pretrained(pretrained)
        self.plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim = self.plm_hidden_size, num_heads = head_nums, batch_first = True
        )
        self.additive_attention = AdditiveAttention(self.plm_hidden_size, additive_hidden_dim)
        self.logger = logger_wrapper("TE")
        
    def forward(self, ids, mask)->torch.Tensor:
        '''
        work flow:
            1. Using the pretrained model to get the vector
            2. multihead attention
            3. additive attention
        return a tensor [batchsize, hiddensize]
        ==================================================
        **NOTE: bs here is not the batchsize, it is the num of condidate news or**
              **the length of the histories, depends on the input.**
        transform flow:
            input   ->      plm       ->     MH_att     ->    add_atten   ->  sum
        (bs, seqlen)->(bs, seqlen, hs)->(bs, seqlen, hs)->(bs, seqlen, hs)->(bs, hs)
        note:
            seqlen means the length of token id sequence.
            hs means hiddensize.
        '''
        # the output of the PLM is always the [batchsize, seqlen, hidden_size]
        V = self.pretrained_textual_model(ids, mask).last_hidden_state
        # then using the MHA to get the self attention and then comprehend the sequence.
        multihead_attention_output, _ = self.multihead_attention(V,V,V)
        # using the additive attention to get the global information.
        
        output = self.additive_attention(multihead_attention_output)
        # using sum to congregate the info of the whole news text.
        output = torch.sum(output, dim = 1)
        return output



class VisionEncoder(nn.Module):
    def __init__(self, pretrained: str="facebook/dinov2-base",
                 num_heads = 8, additive_hidden_dim = 256):
        super().__init__()
        self.pretrained_visual_model = AutoModel.from_pretrained(pretrained)
        self.plm_hidden_size = AutoConfig.from_pretrained(pretrained).hidden_size
        self.num_heads = num_heads
        self.additive_hidden_dim = additive_hidden_dim
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim = self.plm_hidden_size, num_heads = self.num_heads, batch_first = True
        )
        self.additive_attention = AdditiveAttention(self.plm_hidden_size, self.additive_hidden_dim)
        self.logger = logger_wrapper("VE")
        
    def forward(self, encoded_input)->torch.Tensor:
        '''
        work flow:
            1. Using the pretrained model to get the vector
            2. multihead attention
            3. additive attention
        return a tensor [batchsize, hiddensize]
        ==================================================
        transform flow:
            input   ->      pvm       ->     MH_att     ->    add_atten   ->  sum
        (bs, seqlen)->(bs, seqlen, hs)->(bs, seqlen, hs)->(bs, seqlen, hs)->(bs, hs)
        note:
            **bs here is not the batchsize, it is the num of condidate news or**
                **the length of the histories, depends on the input.**
            seqlen means the length of pic patches sequence.
            hs means hiddensize.
        '''
        # the output of the PVM is always the [batchsize, seqlen, hidden_size]
        V = self.pretrained_visual_model(encoded_input).last_hidden_state
        # then using the MHA to get the self attention and then comprehend the sequence.
        multihead_attention_output, _ = self.multihead_attention(V,V,V)
        # using the additive attention to get the global information.
        output = self.additive_attention(multihead_attention_output)
        # using sum to congregate the info of the whole news text.
        output = torch.sum(output, dim = 1)
        return output

class MultiModalEncoder(nn.Module):
    def __init__(self, TextualEncoder, VisualEncoder):
        super().__init__()
        self.TextualEncoder = TextualEncoder
        self.VisualEncoder = VisualEncoder
        self.logger = logger_wrapper("MME")
        
    def forward(self, ids, mask, vis):
        '''
        batch includes the keys:
            textual_encoded_input, visual_encoded_input, labels
        work flow:
            1. using the pretrained model to get the feature.
            2. fusion
        '''
        # work flow 1:
        
        V_text = self.TextualEncoder(ids, mask)
        # self.logger.info("V_text's shape is:",V_text.shape)
        
        V_vision = self.VisualEncoder(vis)
        # self.logger.info("V_vision's shape is:",V_vision.shape)
        
        # work flow 2:
        V_fusion = V_text * V_vision
        return V_fusion
        
        






