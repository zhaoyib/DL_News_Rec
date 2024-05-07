'''
@File      :   dataframe.py
@Time      :   2024/04/25 14:15:45
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   TODO: process the EMPTY_NEWS for img.
'''
import hashlib
import inspect
import json
import pickle
from pathlib import Path
from typing import Callable
from PIL import Image
import pandas as pd
import polars as pl
import torch
import numpy as np
from const.path import CACHE_DIR
from const.path import MIND_SMALL_IMG_DATASET_DIR
from utils.logger import logging

# def _cache_dataframe(fn: Callable) -> Callable:
#     def read_df_function_wrapper(*args: tuple, **kwargs: dict) -> pl.DataFrame:
#         # inspect **kwargs
#         bound = inspect.signature(fn).bind(*args, **kwargs)
#         bound.apply_defaults()

#         d = bound.arguments
#         d["function_name"] = fn.__name__
#         d["path_to_tsv"] = str(bound.arguments["path_to_tsv"])

#         # if file exist in cache path, then load & return it.
#         cache_filename = hashlib.sha256(json.dumps(d).encode()).hexdigest()
#         cache_path = CACHE_DIR / f"{cache_filename}.pth"
#         if cache_path.exists() and (not d["clear_cache"]):
#             with open(cache_path, "rb") as f:
#                 df = pickle.load(f)
#             return df

#         df = fn(*args, **kwargs)

#         cache_path.parent.mkdir(parents=True, exist_ok=True)

#         with open(cache_path, "wb") as f:
#             pickle.dump(df, f)

#         return df

#     return read_df_function_wrapper


# @_cache_dataframe
def read_news_df(path_to_tsv: Path, has_entities: bool = False, clear_cache: bool = False) -> pl.DataFrame:
    '''
    the function will return a news df without the entities.
    '''
    news_df = pd.read_csv(path_to_tsv, sep="\t", encoding="utf8", header=None)
    news_df.columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    news_df = pl.from_dataframe(news_df)
    if has_entities:
        return news_df
    return news_df.drop("title_entities", "abstract_entities")


# @_cache_dataframe
def read_behavior_df(path_to_tsv: Path, clear_cache: bool = False) -> pl.DataFrame:
    '''
    first, it split the impression_str to the list and then split it to news id and 0/1.
    it will be organized to a dict, with format: {'news_id': news_id, 'clicked':0/1}
    finally, the history will be split to list.
    it will return: 
        impression_id: useless in fact.
        user_id: useless in fact as well.
        time: useless too.
        history: a list of historial news id.
        impressions: a list of dict with the format described above.
    '''
    behavior_df = pl.read_csv(path_to_tsv, separator="\t", encoding="utf8-lossy", has_header=False)
    behavior_df = behavior_df.rename(
        {
            "column_1": "impression_id",
            "column_2": "user_id",
            "column_3": "time",
            "column_4": "history_str",
            "column_5": "impressions_str",
        }
    )
    try:
        behavior_df = (
            behavior_df.with_columns((pl.col("impressions_str").str.split(" ")).alias("impression_news_list"))
            .with_columns(
                [
                    pl.col("impression_news_list")
                    .apply(lambda v: [{"news_id": item.split("-")[0], "clicked": int(item.split("-")[1])} for item in v])
                    .alias("impressions")
                ]
            )
            .with_columns([pl.col("history_str").str.split(" ").alias("history")])
            .select(["impression_id", "user_id", "time", "history", "impressions"])
        )
    except Warning as w:
        pass
    logging.info(f"the original behavior dataframe is: {behavior_df[0]}")
    return behavior_df

def batch_text_transform_fn(text: list, tokenizer) -> torch.Tensor:
    '''
    tokenizer should be got by the transformers.AutoTokenizer.from_pretrained(model_name)
    input the text and the tokenizer, return the ids and mask.
    '''
    output = tokenizer(text, return_tensors = "pt", max_length = 30,padding = "max_length", truncation = True)
    input_ids = output["input_ids"]
    attention_mask = output["attention_mask"]
    return input_ids, attention_mask


def batch_img_transform_fn(imgs: list, preprocessor) -> torch.Tensor:
    '''
    preprocessor should be got by the transformers.AutoImageProcessor.from_pretrained(model_name)
    input the img and preprocessor, return the batch for vision model to input.
    '''
    output = preprocessor(imgs)
    return output
    

def build_img_dict():
    '''
    use the projector, make the id be id in small, the value is the 
    '''
    projector = _build_projector()
    img_dict = {}
    for key, value in projector.items():
        img_dict[key] = _read_pic(value)
    img_dict["EMPTY_NEWS_ID"] = Image.fromarray(np.full((195,610,3),255).astype(np.uint8))
    # if input the pics only, it should be 165,200,3
    img_dict[-1] = Image.fromarray(np.full((195,610,3),255).astype(np.uint8))
    return img_dict

def _read_pic(large_id):
    file_path = MIND_SMALL_IMG_DATASET_DIR / (large_id + ".jpg")
    image = Image.open(file_path)
    res = image
    # print(np.array(res).shape)
    # the why do we click used the whole pic to get the visual feature.
    # It makes sense better than input a white pic for the blank ones.
    # res = image.crop((15,15,215,180))
    # don't process it here. control the flow at the pipeline maybe better.
    # if is_image_all_white(res):
    #     res = "None"
    return res

def is_image_all_white(img):
    img_array = np.array(img)
    
    gray_img_array = np.mean(img_array, axis=2)
    
    return np.all(gray_img_array == 255)


def _build_projector():
    '''
    a projector dict, key is the id in MIND Small
    value is the id in MIND.
    '''
    def read_csv_columns(file_path, key_col, value_col):
        data = pd.read_csv(file_path)
        csv_dict = dict(zip(data[key_col], data[value_col]))
        return csv_dict
        
    csv_file = "autodl-tmp/projector.csv"  
    first_col = "nid"  
    last_col = "IM-MIND-INDEX"    
    csv_dict = read_csv_columns(csv_file, first_col, last_col)
    return csv_dict


if __name__ == "__main__":
    _read_pic("N3")