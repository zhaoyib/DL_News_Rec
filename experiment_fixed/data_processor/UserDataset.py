'''
@File      :   UserDataset.py
@Time      :   2024/04/11 16:00:12
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   Create the UserDataset.py
'''

# abs import
from data_processor.BaseDataset import BaseDataset
from utils.load_from_files import Get_user_data
from utils.logger import logger_wrapper
from utils.load_configs import Config
import torch


#please define the transform as : transform = lambda x: transform_function(x)

class UserDataset(BaseDataset):
    def __init__(self, data, transform=None):
        super().__init__(data, transform)
        
    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_UserDataset(configs):
    logger = logger_wrapper("UserDataset",path = configs.logger_path)
    user_data = Get_user_data(configs)
    raw_user_dataset = UserDataset(user_data)
    logger.info("successfully built raw user dataset.")
    return raw_user_dataset