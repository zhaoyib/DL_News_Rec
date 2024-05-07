'''
@File      :   NewsDataset.py
@Time      :   2024/04/11 09:44:10
@LastEdit  :   2024/04/11 11:49:06
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   NewsDataset, Reload the BaseDataset,add get_NewsDataset function
                to easily load the NewsDataset.
'''

# abs import
from data_processor.BaseDataset import BaseDataset
from utils.load_from_files import Get_news_data
from utils.logger import logger_wrapper
from utils.load_configs import Config
import torch


#please define the transform as : transform = lambda x: transform_function(x)

class NewsDataset(BaseDataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)
        
    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_NewsDataset(configs):
    logger = logger_wrapper("NewsDataset",path = configs.logger_path)
    news_data = Get_news_data(configs)
    newsdataset = NewsDataset(news_data)
    logger.info("successfully built newsdataset.")
    return newsdataset