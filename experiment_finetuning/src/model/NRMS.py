'''
@File      :   NRMS.py
@Time      :   2024/04/23 22:32:46
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   create NRMS.py
'''

import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput

class NRMS(nn.Module):
    def __init__(self, user_encoder, news_encoder, d_model, logger, loss_fn = nn.CrossEntropyLoss()) -> None:
        super().__init__()
        self.user_encoder = user_encoder
        self.news_encoder = news_encoder
        self.d_model = d_model
        self.loss_fn = loss_fn
        self.logger = logger

    def forward(self, histories_text, histories_mask, histories_imgs,
               condidate_text, condidate_mask, condidate_imgs, labels):
        '''
        input:
            the batch made by DataLoader is a dict, and then mapped by transformers.Trainer.
            See details at transformers.trainer.Trainer.training_step, it calls self._prepare_inputs,
            which maps the dict input to the seperate args.
            histories_text    : torch.Tensor # ids of the PLM
            histories_mask    : torch.Tensor # mask of the PLM
            histories_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            condidate_text    : torch.Tensor # ids of the PLM
            condidate_mask    : torch.Tensor # mask of the PLM
            condidate_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            label             : torch.Tensor # the clicked position
        work flow:
            1. call the Multimodal_NewsEncoder to encode the condidates.
            2. call the UserEncoder to get the Users' interest.
            3. calculate the similarity betweent the users and the condidates.
        '''
        batch_size, condidate_num, seq_len = condidate_text.size()
        
        condidate_text = condidate_text.view(batch_size * condidate_num, seq_len)
        condidate_mask = condidate_mask.view(batch_size * condidate_num, seq_len)
        # not sure about it. maybe the shape of tensor is not such a shape.
        _, _, channel, height, weight = condidate_imgs.shape
        condidate_imgs = condidate_imgs.view(batch_size * condidate_num, channel, height,weight)
        
        news_condidate_encoded = self.news_encoder(condidate_text, condidate_mask, condidate_imgs)
        
        news_condidate_encoded = news_condidate_encoded.view(batch_size, condidate_num, self.d_model)
        
        news_histories_encoded = self.user_encoder(
            histories_text, histories_mask, histories_imgs, self.news_encoder
        )# now the news_histories_encoded is (batch_size, emb_dim)
        
        news_histories_encoded = news_histories_encoded.unsqueeze(-1)
        # turn to be [batch_size, emb_dim, 1]
        
        output = torch.bmm(
            news_condidate_encoded, news_histories_encoded
        )
        # bmm details see: https://pytorch.org/docs/2.0/generated/torch.bmm.html?highlight=bmm#torch.bmm
        # news_condidate_encoded is [batch_size, condidate_num, emb_dim]
        # news_histories_encoded is [batch_size, emb_dim, 1]
        # it will return a tensor with [batch_size, condidate_num, 1]
        # which means doing matmul for each matrix in the tensor.
        # it means calculate the dot product to get the similarity for each pair of user and condidate.
        output = output.squeeze(-1)
        
        if not self.training:
            return ModelOutput(logits=output, loss = torch.Tensor([-1]), labels=labels)
        
        loss = self.loss_fn(output, labels)
        return ModelOutput(logits=output, loss=loss, labels=labels)