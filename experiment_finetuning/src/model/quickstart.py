'''
@File      :   quickstart.py
@Time      :   2024/04/29 11:12:56
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   get the net quickly.
'''
from model.NRMS import NRMS
from model.Multimodal_NewsEncoder import MultiModalEncoder, TextEncoder, VisionEncoder
from model.UserEncoder import UserEncoder
from utils.logger import logger_wrapper

def get_NRMS():
    text_encoder = TextEncoder()
    vis_encoder = VisionEncoder()
    news_encoder = MultiModalEncoder(text_encoder, vis_encoder)
    user_encoder = UserEncoder(d_model = 768)
    logger = logger_wrapper("NRMS")
    net = NRMS(user_encoder, news_encoder, d_model = 768, logger = logger)
    return net

if __name__ == "__main__":
    net = get_NRMS()
    print(net)
    from data_process.dataframe import _read_pic
    img = _read_pic("N3")
    