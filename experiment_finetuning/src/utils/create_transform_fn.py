'''
@File      :   create_transform_fn.py
@Time      :   2024/04/26 00:41:19
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   input a tokenizer for the text_transform_fn
               alse a preprocessor for the img_transform_fn
'''

from typing import Callable
from transformers import PreTrainedTokenizer
from transformers import AutoImageProcessor
import torch


def create_text_transform_fn_from_pretrained_tokenizer(
    tokenizer: PreTrainedTokenizer, max_length: int, padding: bool = True
) -> Callable[[list[str]], torch.Tensor]:
    def text_transform(texts: list[str]) -> torch.Tensor:
        return tokenizer(texts, return_tensors="pt", max_length=64, padding="max_length", truncation=True)["input_ids"]

    return text_transform


def create_img_transform_fn_from_pretrained_processor(
    processor
)