'''
@File      :   VisionFeature.py
@Time      :   2024/03/27 14:42:32
@LastEdit  :   2024/03/28 09:56:33
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
import torch
import numpy as np
from numpy import ndarray
from typing import List,Dict,Tuple,Type,Union
from transformers import AutoModel, AutoImageProcessor
from utils.load_configs import Config
from utils.logger import logger_wrapper


class VisionModel():
    def __init__(self,configs:Config) -> None:
        '''
        Initial the VisionFeature Extractor.

        parameter:
            configs : class Config, load by utils.load_configs.py
                        ->Config().read_from_json(JSON_path)
                it will return a configs instance.
                parameters can be reached by using operator "."
        return:
            None
        '''
        self.access_token = configs.model_set.get("access_token","")
        self.model_name = configs.model_set.get("visual_model_name","")
        self.logger_path = configs.logger_path
        self.device = configs.device[0]
        self.num_gpus = configs.device[1]
        self.model = AutoModel.from_pretrained(self.model_name,use_auth_token=self.access_token)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_auth_token=self.access_token)
        self.logger = logger_wrapper("Visual Feature Extractor",path=self.logger_path)
        self.logger.info(f"Successfully load {self.model_name} from hugging face")
        self.model = self.model.to(self.device)
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t")
        self.batch_size = configs.batch_size

    def encode(self, data:list, 
               return_numpy:bool= False,
               **kwargs):
        '''
        encode the images in data to vector.

        parameters:
            data        : list of the inputs, elements are tuple, with elements:
                        (news_id, title, abstract, img). once a batch.
            return_numpy:  what kind of data to return, if false, return [(str,tensor)].
        return:
            embeddings  : a list with tuple [(news_id,visual_feature)]          
        '''
        news_id = [news[0] for news in data]
        imgs = [news[3] for news in data]
        with torch.no_grad():
            embeddings = []
            inputs = self.processor(imgs)
            #call the ViTImageProcessor.prerocess 
            #(overdrive the preprocess in BaseImageProcessor)
            #inputs = BatchFeature(data) = BaseBatchFeature
            inputs_on_device = {k:torch.tensor(np.array(v)).to(self.device) for k,v in inputs.items()}
            outputs = self.model(**inputs_on_device)
            last_hidden_state = outputs["last_hidden_state"]
            pooler_output = outputs["pooler_output"]
            #according to the paper Better plain ViT baselines for ImageNet 1k
            #https://arxiv.org/abs/2205.01580
            #they replaced the use of CLS token by average pooling and they got better results.
            #last_hidden_state with tensor shape [batchsize,257,768]
            #pooler_output with tensor shape [batchsize,768]
            embeddings = last_hidden_state.mean(dim=1)
            # reference https://discuss.huggingface.co/t/last-hidden-state-vs-pooler-output-in-clipvisionmodel/26281
        embedding_collection = list(zip(news_id,embeddings))
        return embedding_collection