'''
@File      :   BaseDataloader.py
@Time      :   2024/04/11 11:51:16
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   Create the Dataloader.py
'''
from torch.utils.data import DataLoader, Dataset

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0, **kwargs):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs)