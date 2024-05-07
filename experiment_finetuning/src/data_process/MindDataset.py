'''
@File      :   MindDataset.py
@Time      :   2024/04/25 09:41:34
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   train and dev dataset respectively.
'''

'''
Quick Start:
    see quick_start.py
    call the function "Build_Train_DataLoader()" or the "Build_Dev_DataLoader()"
    to get the instance of two DataLoader for train or dev.
'''
import random
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

EMPTY_NEWS_ID, EMPTY_IMPRESSION_IDX = "EMPTY_NEWS_ID", -1

class MindTrainDataset(Dataset):
    def __init__(
        self, 
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        imgs_dict: dict,
        tokenizer,
        preprocessor,
        batch_text_transform_fn,
        batch_img_transform_fn,
        npratio: int,
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        '''
        NOTE:
            This part is connected tightly with the dataframe.py, please check the dataframe.py first.
        work flow:
            1. load the configs from the input.
            2. extract the related data.
        '''
        # super().__init__()
        # load the configs first.
        self.behavior_df : pl.DataFrame = behavior_df
        self.news_df : pl.DataFrame = news_df
        self.batch_text_transform_fn : Callable[[list[str]], torch.Tensor] = batch_text_transform_fn
        self.batch_img_transform_fn : Callable[np.ndarray, torch.Tensor] = batch_img_transform_fn
        self.npratio: int = npratio
        self.history_size: int = history_size
        self.device: torch.device = device
        # the news_id and news_img projector
        self.imgs_dict: dict = imgs_dict
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        
        # extract the related data:
        self.behavior_df = self.behavior_df.with_columns(
            [
                pl.col("impressions")
                .apply(lambda v : [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 1])
                .alias("clicked_idxes"),
                pl.col("impressions")
                .apply(lambda v: [i for i, imp_item in enumerate(v) if imp_item["clicked"] == 0])
                .alias("non_clicked_idxes"),
            ]
        )
        
        # define the news_id and news_text projector
        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["title"].item() for i in range(len(self.news_df))
        }
        
        # define the empty padding projector
        self.__news_id_to_title_map[EMPTY_NEWS_ID] = ""
            

    def __getitem__(self, behavior_idx: int) -> dict:
        '''
        NOTE: it will return ONE user's behavior.
        get a user's behavior, return a dict include:
            histories_text    : torch.Tensor # ids of the PLM
            histories_mask    : torch.Tensor # mask of the PLM
            histories_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            condidate_text    : torch.Tensor # ids of the PLM
            condidate_mask    : torch.Tensor # mask of the PLM
            condidate_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            labels            : torch.Tensor # the clicked position
        work flow:
            1. get the behavior responding to the behavior_idx input.
            2. using the self.__news_id_to_title_map to get history news.
            3. undersample the neg_impressions and the pos. using self.__sampling_negative
            PERIOD SUMMARY: 
                now we get the news ids to load for one user.
                then we need to convert the ids to text and img.
            4. convert ids to text
            5. convert ids to img
            6. delete the white img and build the reference between id and img pix value.
            7. congregate the values
        '''
        # get the behavior responding to the behavior_idx input.
        behavior_item = self.behavior_df[behavior_idx]

        # extract the news ids first.
        # using the bracket to execute the "if" first.
        # like the :? calculator in cpp
        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )

        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}

        # a list of dict.
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )
        
        pos_idxes, neg_idxes = (
            behavior_item["clicked_idxes"].to_list()[0],
            behavior_item["non_clicked_idxes"].to_list()[0],
        )

        # sampling the neg_idxes, using the undersampling method.
        # the npratio is used here, npratio = sampled_neg_count : sampled_pos_count.
        # for the DataLoader will call __getitem__ each epoch, so the undersampling\
        # will return the different samples for different epoches.
        # it will enhance model's generalization ability.
        sampled_pos_idxes = random.sample(pos_idxes, 1)
        sampled_neg_idxes = self.__sampling_negative(neg_idxes, self.npratio)


        sampled_impression_idxes = sampled_pos_idxes + sampled_neg_idxes
        random.shuffle(sampled_impression_idxes)
        # period 1 finished.
        # period 4:
        # 4.1 extract the impression id and history ids respectively.
        sampled_impressions = impressions[sampled_impression_idxes]
        condidate_news_ids = [imp_item["news_id"] for imp_item in sampled_impressions]
        labels = [imp_item["clicked"] for imp_item in sampled_impressions]
        histories_news_ids = history[-1 * self.history_size :]
        if len(history) < self.history_size:
            histories_news_ids = histories_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(histories_news_ids))

        # 4.2 convert ids to context.
        condidate_news_title = [self.__news_id_to_title_map[news_id] for news_id in condidate_news_ids]
        histories_news_title = [self.__news_id_to_title_map[news_id] for news_id in histories_news_ids]

        # 4.2 convert context to tensor and get mask btw.
        condidate_text,condidate_mask=self.batch_text_transform_fn(condidate_news_title, self.tokenizer)
        histories_text,histories_mask=self.batch_text_transform_fn(histories_news_title, self.tokenizer)

        # convert the labels to tensor by the way.
        labels_tensor = torch.Tensor(labels).argmax()

        # 4.3 get the imgs and then convert them to tensor.
        condidate_imgs = [self.imgs_dict[news_id] for news_id in condidate_news_ids]
        histories_imgs = [self.imgs_dict[news_id] for news_id in histories_news_ids]

        # convert to tensor.
        condidate_imgs = self.batch_img_transform_fn(condidate_imgs,self.preprocessor)
        histories_imgs = self.batch_img_transform_fn(histories_imgs, self.preprocessor)
        
        # see details at: https://huggingface.co/docs/transformers/en/main_classes
        #/image_processor#transformers.BatchFeature.tensor_type
        histories_imgs.convert_to_tensors(tensor_type="pt")
        condidate_imgs.convert_to_tensors(tensor_type="pt")
        # print(type(histories_imgs['pixel_values'])) tensor
        histories_imgs = histories_imgs['pixel_values']
        condidate_imgs = condidate_imgs['pixel_values']
        
        assert torch.is_tensor(histories_imgs), f"imgs are not tensor, with type{type(histories_imgs)}"
        assert torch.is_tensor(histories_text), f"text are not tensor, with type{type(histories_text)}"
        assert torch.is_tensor(histories_mask), f"mask are not tensor, with type{type(histories_mask)}"
        assert torch.is_tensor(condidate_imgs), f"imgs are not tensor, with type{type(condidate_imgs)}"
        assert torch.is_tensor(condidate_text), f"text are not tensor, with type{type(condidate_text)}"
        assert torch.is_tensor(condidate_mask), f"mask are not tensor, with type{type(condidate_mask)}"
        assert torch.is_tensor(labels_tensor), f"labels are not tensor, with type{type(labels_tensor)}"
        
        return {
            "histories_text" : histories_text,
            "histories_mask" : histories_mask,
            "histories_imgs" : histories_imgs,
            "condidate_text" : condidate_text,
            "condidate_mask" : condidate_mask,
            "condidate_imgs" : condidate_imgs,
            "labels"         : labels_tensor
        }


    # **work for the __getitem__()**
    def __sampling_negative(self, neg_idxes: list, npratio:int)-> list:
        '''
        add a padding process only.
        '''
        if len(neg_idxes) < npratio:
            # padding with the EMPTY_IMPRESSION_IDX
            return neg_idxes + [EMPTY_IMPRESSION_IDX] * (npratio - len(neg_idxes))
        
        return random.sample(neg_idxes, self.npratio)


    def __len__(self) -> int:
        # basic method for the dataset in pytorch.
        return len(self.behavior_df)


# it won't sample.
class MindDevDataset(Dataset):
    def __init__(
        self, 
        behavior_df: pl.DataFrame,
        news_df: pl.DataFrame,
        imgs_dict: dict,
        tokenizer,
        preprocessor,
        batch_text_transform_fn,
        batch_img_transform_fn,
        history_size: int,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        '''
        work flow:
            1. load the configs from the input.
            2. extract the related data.
            3. define the projector, include:
                3.1 news_id and news_text projector
                3.2 news_id and news_img projector
                3.3 empty padding projector
        entailed function:
            self._MIND_id_projector(MIND_small_news_id) -> MIND_large_news_id.
            self._build_projector() -> a dict combine the small id and large id.
        '''
        # super().__init__()
        # load the configs first.
        self.behavior_df : pl.DataFrame = behavior_df
        self.news_df : pl.DataFrame = news_df
        self.batch_text_transform_fn : Callable[[list[str]], torch.Tensor] = batch_text_transform_fn
        self.batch_img_transform_fn : Callable[np.ndarray, torch.Tensor] = batch_img_transform_fn
        self.history_size: int = history_size
        self.device: torch.device = device
        self.imgs_dict: dict = imgs_dict
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        
        # define the news_id and news_text projector
        self.__news_id_to_title_map: dict[str, str] = {
            self.news_df[i]["news_id"].item(): self.news_df[i]["title"].item() for i in range(len(self.news_df))
        }
        
        # define the empty padding projector
        self.__news_id_to_title_map[EMPTY_NEWS_ID] = ""
    
            
    def __getitem__(self, behavior_idx: int) -> dict:
        '''
        NOTE: it will return ONE user's behavior.
        get a user's behavior, return a dict include:
            histories_text    : torch.Tensor # ids of the PLM
            histories_mask    : torch.Tensor # mask of the PLM
            histories_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            condidate_text    : torch.Tensor # ids of the PLM
            condidate_mask    : torch.Tensor # mask of the PLM
            condidate_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            labels            : torch.Tensor # the clicked position
        work flow:
            1. get the behavior responding to the behavior_idx input.
            2. using the self.__news_id_to_title_map to get history news.
            PERIOD SUMMARY: 
                now we get the news ids to load for one user.
                then we need to convert the ids to text and img.
            3. convert ids to text
            4. convert ids to img
            5. delete the white img and build the reference between id and img pix value.
            6. congregate the values
        '''
        # get the behavior responding to the behavior_idx input.
        behavior_item = self.behavior_df[behavior_idx]

        # extract the news ids first.
        # using the bracket to execute the "if" first.
        # like the :? calculator in cpp
        history: list[str] = (
            behavior_item["history"].to_list()[0] if behavior_item["history"].to_list()[0] is not None else []
        )

        EMPTY_IMPRESSION = {"news_id": EMPTY_NEWS_ID, "clicked": 0}

        # a list of dict.
        # list + list will be extended.
        impressions = np.array(
            behavior_item["impressions"].to_list()[0] + [EMPTY_IMPRESSION]
        )
        
        # period 1 finished.
        # period 4:
        # 4.1 extract the impression id and history ids respectively.
        condidate_news_ids = [imp_item["news_id"] for imp_item in impressions]
        labels = [imp_item["clicked"] for imp_item in impressions]
        histories_news_ids = history[-1 * self.history_size :]
        if len(history) < self.history_size:
            histories_news_ids = histories_news_ids + [EMPTY_NEWS_ID] * (self.history_size - len(histories_news_ids))

        # 4.2 convert ids to context.
        condidate_news_title = [self.__news_id_to_title_map[news_id] for news_id in condidate_news_ids]
        histories_news_title = [self.__news_id_to_title_map[news_id] for news_id in histories_news_ids]

        # 4.2 convert context to tensor and get mask btw.
        condidate_text,condidate_mask=self.batch_text_transform_fn(condidate_news_title, self.tokenizer)
        histories_text,histories_mask=self.batch_text_transform_fn(histories_news_title, self.tokenizer)

        # convert the labels to tensor by the way.
        labels_tensor = torch.Tensor(labels)

        # 4.3 get the imgs and then convert them to tensor.
        # print(condidate_news_ids[0])
        condidate_news_img = [self.imgs_dict[news_id] for news_id in condidate_news_ids]
        histories_news_img = [self.imgs_dict[news_id] for news_id in histories_news_ids]

        # convert to BatchFeature.
        condidate_imgs = self.batch_img_transform_fn(condidate_news_img,self.preprocessor)
        histories_imgs = self.batch_img_transform_fn(histories_news_img, self.preprocessor)
        # convert to pytorhc tensor.
        # see details at: https://huggingface.co/docs/transformers/en/main_classes
        #/image_processor#transformers.BatchFeature.tensor_type
        histories_imgs.convert_to_tensors(tensor_type="pt")
        condidate_imgs.convert_to_tensors(tensor_type="pt")
        # print(type(histories_imgs['pixel_values'])) tensor
        histories_imgs = histories_imgs['pixel_values']
        condidate_imgs = condidate_imgs['pixel_values']
        
        assert torch.is_tensor(histories_imgs), f"imgs are not tensor, with type{type(histories_imgs)}"
        assert torch.is_tensor(histories_text), f"text are not tensor, with type{type(histories_text)}"
        assert torch.is_tensor(histories_mask), f"mask are not tensor, with type{type(histories_mask)}"
        assert torch.is_tensor(condidate_imgs), f"imgs are not tensor, with type{type(condidate_imgs)}"
        assert torch.is_tensor(condidate_text), f"text are not tensor, with type{type(condidate_text)}"
        assert torch.is_tensor(condidate_mask), f"mask are not tensor, with type{type(condidate_mask)}"
        assert torch.is_tensor(labels_tensor), f"labels are not tensor, with type{type(labels_tensor)}"
        
        return {
            "histories_text" : histories_text,
            "histories_mask" : histories_mask,
            "histories_imgs" : histories_imgs,
            "condidate_text" : condidate_text,
            "condidate_mask" : condidate_mask,
            "condidate_imgs" : condidate_imgs,
            "labels"         : labels_tensor
        }

    def __len__(self) -> int:
        # basic method for the dataset in pytorch.
        return len(self.behavior_df)