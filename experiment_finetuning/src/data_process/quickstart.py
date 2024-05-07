'''
@File      :   quickstart.py
@Time      :   2024/04/25 14:13:53
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   quickstart for data process.
'''

from data_process.MindDataset import MindTrainDataset, MindDevDataset
from data_process.dataframe import batch_text_transform_fn, batch_img_transform_fn
from data_process.dataframe import read_behavior_df, read_news_df, build_img_dict
from utils.logger import logger_wrapper

from const.path import MIND_SMALL_VAL_DATASET_DIR
from const.path import MIND_SMALL_TRAIN_DATASET_DIR
from const.path import ACCESS_TOKEN
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor


def Build_Dev_DataLoader():
    logger = logger_wrapper("Dev Dataset")

    logger.info("Loading Data...")
    behavior_df = read_behavior_df(MIND_SMALL_VAL_DATASET_DIR / "behaviors.tsv")
    news_df = read_news_df(MIND_SMALL_VAL_DATASET_DIR / "news.tsv")
    imgs_dict = build_img_dict()
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", token=ACCESS_TOKEN)
    preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", token=ACCESS_TOKEN)
    logger.info("Building Dataset...")
    dev_dataset = MindDevDataset(behavior_df, news_df, imgs_dict, tokenizer,
                                 preprocessor,batch_text_transform_fn,
                                 batch_img_transform_fn, history_size = 50)
    # the following only used in test. not used in fact.
    dev_dataloader = DataLoader(dev_dataset, batch_size = 1, shuffle = True)
    logger.info("Succeed!")
    logger.info('''get a user's behavior, return a dict include:
            histories_text    : torch.Tensor # ids of the PLM
            histories_mask    : torch.Tensor # mask of the PLM
            histories_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            condidate_text    : torch.Tensor # ids of the PLM
            condidate_mask    : torch.Tensor # mask of the PLM
            condidate_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            label             : torch.Tensor # the clicked position''')
    # for example in dev_dataloader:
    #     logger.info(f"an example of batch{example.keys()}")
    #     foo = dir(example["histories_imgs"])
    #     logger.info(f"the imgs include the following parts: {foo}")
    #     break
    return dev_dataset, dev_dataloader

def Build_Train_DataLoader():
    logger = logger_wrapper("Train Dataset")

    logger.info("Loading Data...")
    behavior_df = read_behavior_df(MIND_SMALL_TRAIN_DATASET_DIR / "behaviors.tsv")
    news_df = read_news_df(MIND_SMALL_TRAIN_DATASET_DIR / "news.tsv")
    imgs_dict = build_img_dict()
    preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224", token=ACCESS_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", token=ACCESS_TOKEN)
    
    logger.info("Building Dataset...")
    train_dataset = MindTrainDataset(behavior_df, news_df, imgs_dict, tokenizer,
                                 preprocessor,batch_text_transform_fn,
                                 batch_img_transform_fn, npratio = 4, history_size = 50)
    # the following only for test, not used in fact.
    train_dataloader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
    logger.info("Succeed!")
    logger.info('''get a user's behavior, return a dict include:
            histories_text    : torch.Tensor # ids of the PLM
            histories_mask    : torch.Tensor # mask of the PLM
            histories_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            condidate_text    : torch.Tensor # ids of the PLM
            condidate_mask    : torch.Tensor # mask of the PLM
            condidate_imgs    : torch.Tensor # pix value of the pic. input of the VLM
            label             : torch.Tensor # the clicked position''')
    # for example in train_dataloader:
    #     logger.info(f"an example of batch{example}")
    #     break
    return train_dataset, train_dataloader







