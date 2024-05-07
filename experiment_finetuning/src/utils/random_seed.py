'''
@File      :   random_seed.py
@Time      :   2024/04/25 16:43:32
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   set the random seed.
'''



import os
import random

import numpy as np
import torch


def set_random_seed(random_seed: int = 42) -> None:
    random.seed(random_seed)
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True