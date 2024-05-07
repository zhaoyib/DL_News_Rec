'''
@File      :   path.py
@Time      :   2024/04/25 16:40:37
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   auto generate the path of folder.
'''

from datetime import datetime
from pathlib import Path


def generate_folder_name_with_timestamp(path_prefix: Path, timestamp: datetime = datetime.now()) -> Path:
    date = Path(timestamp.strftime("%Y-%m-%d"))
    time = Path(timestamp.strftime("%H-%M-%S"))
    return path_prefix / date / time