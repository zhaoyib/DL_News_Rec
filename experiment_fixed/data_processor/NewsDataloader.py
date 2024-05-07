'''
@File      :   NewsDataloader.py
@Time      :   2024/04/11 13:51:22
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   Create the NewsDataloader.py
'''
from torch.utils.data import DataLoader, Dataset
from data_processor.BaseDataloader import BaseDataLoader
from utils.logger import logger_wrapper

class NewsDataLoader(BaseDataLoader):
    def __init__(self, dataset, configs, **kwargs):
        batch_size, logger_path = self.parse_configs(configs)
        self.logger = logger_wrapper("NewsDataLoader",path = logger_path)
        super().__init__(dataset, batch_size=batch_size, shuffle=False,num_workers = 0, **kwargs)

    def parse_configs(self,configs):
        batch_size = configs.batch_size
        logger_path = configs.logger_path
        return batch_size,logger_path