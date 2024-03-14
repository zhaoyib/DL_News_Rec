'''
@File      :   visual_model.py
@Time      :   2024/03/14 10:26:40
@LastEdit  :   2024/03/14 14:51:30
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
'''
import torch
from tqdm import tqdm
from numpy import ndarray
from typing import List,Dict,Tuple,Type,Union
from transformers import ViTImageProcessor, ViTModel
from logger import logger_wrapper
logger = logger_wrapper()
access_token = "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"

class Visual_Model():
    def __init__(self, lr:float,model_name:str= "google/vit-base-patch16-224-in21k",
                 device:str=None,**kwargs) -> None:
        '''
        Init a Model To Process the Text Data.

        parameter: lr : learning rate of the model.
                   model_name : the pretrained model chosen to use.
        '''
        self.lr = lr
        self.processor = ViTImageProcessor.from_pretrained(model_name, token=access_token)
        self.model = ViTModel.from_pretrained(model_name, token=access_token)
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

    def encode(self,images:dict,batch_size:int= 64,
               return_numpy:bool= True, enable_tqdm:bool= True,
               **kwargs):
        '''
        encode the inputs images to vector.

        parameters:
            images      : dict only, the value is image pixels, key is image id
            batch_size  : how many images processed together.
            return_numpy: what kind of data to return, if false, return tensor.
            enable_tqdm : will it be interesting. I like tqdm.
        return:
            embeddings  : a dict include the following key-value pairs
                key     : value
                news_id : a numpy list or a tensor representing the meaning of passages.
            it will be return in a dict.
            you can use embeddings.get() method to query the representation of passage you want.
        '''
        keys = images.keys()
        pics = images.values()
        with torch.no_grad():
            embeddings_collection = {}
            for image_id in tqdm(0, len(pics), batch_size, desc = "Extract Embeddings", disable= not enable_tqdm):
                image_batch = pics[image_id:min(image_id+batch_size,len(pics))]
                id_batch = keys[image_id:min(image_id+batch_size,len(pics))]
                inputs = self.processor(
                    image_batch,

                )
                inputs_on_device = {k:v.to(self.device) for k,v in inputs.items()}
                outputs = self.model(**inputs_on_device,return_dict = True)
                embeddings = outputs["pooler_output"]
                embeddings_collection.update(dict(zip(id_batch,embeddings)))
        if return_numpy and not isinstance(embeddings_collection,ndarray):
            for key,value in embeddings_collection.items():
                embeddings_collection[key] = value.numpy()
        return embeddings_collection

if __name__ == "__main__":
    model = Visual_Model(lr=0.01)
    print(model.model)