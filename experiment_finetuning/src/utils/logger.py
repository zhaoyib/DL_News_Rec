'''
@File    :   logger.py
@Time    :   2024/03/11 11:46:05
@Author  :   YiboZhao 
@Version :   1.0
@Site    :   https://github.com/zhaoyib
'''

import logging
from logging import FileHandler

def logger_wrapper(name='News Recommendation System', path=None):
    logging.basicConfig(format='%(asctime)s - [%(levelname)s] - %(name)s ->>> %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(name)
    
    if path:
        file_handler = FileHandler(path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s ->>> %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger