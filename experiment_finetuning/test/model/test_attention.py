'''
@File      :   test_attention.py
@Time      :   2024/04/25 16:48:48
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   test the additive attention
'''

import torch

from recommendation.nrms.AdditiveAttention import AdditiveAttention


def test_additive_attention() -> None:
    batch_size, seq_len, emb_dim, hidden_dim = 20, 10, 30, 5
    attn = AdditiveAttention(emb_dim, hidden_dim)
    input = torch.rand(batch_size, seq_len, emb_dim)
    assert tuple(attn(input).shape) == (batch_size, seq_len, emb_dim)