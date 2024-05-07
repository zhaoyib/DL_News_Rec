'''
@File      :   TextFeature.py
@Time      :   2024/03/21 16:50:07
@LastEdit  :   2024/03/27 14:42:24
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
import torch
import numpy
from numpy import ndarray
from typing import List,Dict,Tuple,Type,Union
from transformers import AutoModel,AutoTokenizer
from utils.logger import logger_wrapper
from utils.load_configs import Config

class TextModel():
    def __init__(self, configs:Config):
        '''
        Initial the TextFeature Extractor.

        parameter:
            configs : class Config, load by utils.load_configs.py
                        ->Config().read_from_json(JSON_path)
                it will return a configs instance.
                parameters can be reached by using operator "."
        return:
            None
        '''
        self.access_token = configs.model_set.get("access_token","")
        self.model_name = configs.model_set.get("text_model_name","")
        self.logger_path = configs.logger_path
        self.device = configs.device[0]
        self.num_gpus = configs.device[1]
        self.model = AutoModel.from_pretrained(self.model_name,use_auth_token=self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,use_auth_token=self.access_token)
        self.logger = logger_wrapper("Text Feature Extracter",path=self.logger_path)
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
        encode the text to feature vector with 768 dimensions.
        parameter:
            data: list of the inputs, elements are tuple, with elements:
                (news_id, title, abstract,img). once a batch.
            return_numpy: a bool to determin whether return a numpy ndarray
            enable_tqdm : a bool to determin whether using tqdm
        return :
            a list of tuple, including:
                (news_id, embedding)
        '''
        if self.num_gpus>1 :
            batch_size = batch_size * self.num_gpus
        assert isinstance(data,list), "Please Input the Data with List Structure.\
            Element should be tuple with (news_id, news_title, news_abstract, image)."

        news_id = [tup[0] for tup in data]
        title_list = [tup[1] for tup in data]
        abs_list = [tup[2] for tup in data]
        #input_list = [s1 + s2 for s1, s2 in zip(title_list,abs_list)]
        input_list = title_list
        with torch.no_grad():
            inputs = self.tokenizer(
                input_list,
                padding = True,
                truncation = True,
                max_length = 512,
                return_tensors = "pt"
            )
            inputs_on_device = {k:v.to(self.device) for k,v in inputs.items()}
            outputs = self.model(**inputs_on_device, return_dict = True)
            #using cls pooler.
            embeddings = outputs.last_hidden_state[:,0]
            if return_numpy and not isinstance(embeddings,ndarray):
                for i,embedding in enumerate(embeddings):
                    embeddings[i] = embedding.numpy()
            
            embedding_collection = list(zip(news_id,embeddings))
        return embedding_collection