'''
@File      :   text_model.py
@Time      :   2024/03/14 09:56:10
@LastEdit  :   2024/03/14 14:51:41
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
import torch
import numpy
from tqdm import tqdm
from numpy import ndarray
from typing import List,Dict,Tuple,Type,Union
from transformers import AutoModel,AutoTokenizer
from logger import logger_wrapper
logger = logger_wrapper()
access_token = "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"

class Text_Model():
    def __init__(self, lr:float, model_name:str = "skimai/spanberta-base-cased",
                 pooler:str= "cls", device:str=None,**kwargs) -> None:
        '''
        Init a Model To Process the Text Data.

        parameter: lr : learning rate of the model.
                   pooler : method of pooler, cls or mean.
        '''
        self.lr = lr
        self.pooler = pooler
        self.model = AutoModel.from_pretrained(model_name, token = access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,token = access_token)
        logger.info(f"loading from {model_name}")

        #detact GPU devices. no need to change it.
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")
        #end of detact and set GPU devices.

        self.model = self.model.to(self.device)#deploy the model to GPU(if there is a GPU)
        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)#multi-GPU, using Parallel method.
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t")
        pass

    def encode(self,sentences:dict,batch_size:int= 64,
               max_length:int= 512, return_numpy:bool= True,
               enable_tqdm:bool= True, **kwargs):
        '''
        encode the inputs sentences to vector.

        parameters:
            sentences   : dict only, the value of dict should be text.
            batch_size  : how many passages processed together.
            max_length  : how long the passages can be.
            return_numpy: what kind of data to return.
            enable_tqdm : will it be interesting. I like tqdm.
        return:
            embeddings  : a dict include the following key-value pairs:
                key       : value
                news_id : a numpy list or a tensor representing the meaning of passages.
            it will be return in a dict.
            you can use embeddings.get() method to query the representation of passage you want.
        '''
        if self.num_gpus>1 :
            batch_size = batch_size * self.num_gpus
        assert isinstance(sentences,dict), "Please Input the Sentences with Dict Structure. Keys Should be IDs."
        
        keys = sentences.keys()
        texts = sentences.values()
        with torch.no_grad():
            embeddings_collection = {}
            for sentence_id in tqdm(0, len(texts), batch_size, desc= "Extract embeddings",disable= not enable_tqdm):
                sentence_batch = texts[sentence_id:min(sentence_id+batch_size,len(texts))]
                id_batch = keys[sentence_id:min(sentence_id+batch_size,len(texts))]
                inputs = self.tokenizer(
                    sentence_batch,
                    padding = True,
                    truncation = True,
                    max_length = 512,
                    return_tensors = "pt"
                )
                inputs_on_device = {k:v.to(self.device) for k,v in inputs.items()}
                outputs = self.model(**inputs_on_device,return_dict = True)
                #embeddings_collection.append(embeddings.cpu())
                if self.pooler == 'cls':
                    embeddings = outputs.last_hidden_state[:,0]#it will be tensor with shape [batch_size,768]
                    #which means all the meaning will be collected to the first token 'cls'
                elif self.pooler == 'mean':
                    attention_mask = inputs_on_device['attention_mask']
                    last_hidden = outputs.last_hidden_state# it will be tensor with shape [batch_size,length,768]
                    embeddings = ((last_hidden * attention_mask.unsqueeze(-1).float()).sum(1)
                                  / attention_mask.sum(-1).unsqueeze(-1))
                else:
                    raise NotImplementedError
                embeddings_collection.update(dict(zip(id_batch,embeddings)))
        if return_numpy and not isinstance(embeddings_collection,ndarray):
            for key,value in embeddings_collection.items():
                embeddings_collection[key] = value.numpy()
        
        return embeddings_collection
        

if __name__ == "__main__":
    model = Text_Model(lr=0.01)
    print(model.model)