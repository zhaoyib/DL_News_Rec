'''
@File      :   BaseDataset.py
@Time      :   2024/04/11 09:40:49
@LastEdit  :   2024/04/11 11:49:06
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   Define the basedataset for the following code to reload.
'''
#please define the transform as : transform = lambda x: transform_function(x)

from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (list): A list of data samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
