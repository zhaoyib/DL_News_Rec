'''
@File      :   News_Embedding_Pipeline.py
@Time      :   2024/03/28 10:13:07
@LastEdit  :   2024/04/10 15:17:17
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   BLIP not available still,but the user batch has been built.
'''
import torch
import pickle as pkl
from math import ceil
from tqdm import tqdm
from lavis.models import load_model_and_preprocess
from feature_extract.TextFeature import TextModel
from feature_extract.VisionFeature import VisionModel
from typing import List,Dict,Tuple,Type,Union
from data_processor.NewsDataloader import NewsDataLoader
from utils.load_configs import Config
from utils.load_from_files import *

class News_Embedding_Pipeline():
    def __init__(self, config) -> None:
        self.using_lavis = config.using_lavis
        self.emb_path = config.emb_path
        if not self.using_lavis:
            self.fusion_method = config.fusion_method #has been checked in configs class
            self.text_model = TextModel(config)
            self.visual_model = VisionModel(config)
        else:
            self.model = config.lavis_model_name
            self.device = config.device[0]
            self.model_type = config.model_type


    def encode(self,data:list):
        '''
        encode the batch built by Dataloader.

        parameter:
            data: list of the inputs, elements are tuple, with elements:
                (news_id, title, abstract,img). once a batch.
        return:
            fusion_embedding : 
                a list of embeddings if using cross attention:
                    [(news_id,text_embedding,visual_embedding)] and then train with CA module.
                if not using cross attention:
                    [(news_id,fusion_embedding)] and then can be saved as static features.
        '''
        batch_first_data = [tuple(batch[i] for batch in data) for i in range(len(data[0]))]
        data = batch_first_data
        if not self.using_lavis:
            fusion_method = self.fusion_method
            text_embedding = self.text_model.encode(data)
            visual_embedding = self.visual_model.encode(data)
            if fusion_method in ["Hadamard Product","HP"]:
                fusion_embedding = self._hadamard_product(text_embedding,visual_embedding)
            if fusion_method in ["Pointwise Add","PA"]:
                fusion_embedding = self._pointwise_add(text_embedding,visual_embedding)
            if fusion_method in ["Concatenate","C"]:
                fusion_embedding = self._concatenate(text_embedding,visual_embedding)
            if fusion_method in ["Cross Attention","CA"]:
                ids = [item[0] for item in text_embedding]
                text_emb = [item[1] for item in text_embedding]
                visual_emb = [item[1] for item in visual_embedding]
                embeddings = list(zip(text_emb,visual_emb))
                fusion_embedding = dict(zip(ids,embeddings))
            return fusion_embedding
        else:
            model, vis_processor, txt_processor = load_model_and_preprocess(
                name = self.model, model_type = self.model_type, 
                is_eval = True, device = self.device
            )
            #download the model_base_capfilt_large.pth to user/.cache/torch/hub/checkpoints/ defaulted.
            print(model)
            print(vis_processor)
            print(txt_processor)


    def save_encode(self,dataloader):
        '''
        save the embeddings with pkl format.
        all tuples a file.
        '''
        path = self.emb_path
        result = {}
        for batch in tqdm(dataloader, desc="Saving Encode"):
            emb = self.encode(batch)
            result.update(emb)
        with open(path,'wb') as f:
            pkl.dump(result,f)

    def _hadamard_product(self,text_embedding, visual_embedding):
        '''
        hadamard product, which means the Pointwise Product.
        require the two embeddings with same dimension.

        parameter:
            text_embedding  : the result of text_model.encode(), a list with [(news_id,embedding)]
            visual_embedding: the result of visual_embedding.encode, a list with [(news_id,embedding)]
        return  :
            fusion_embedding: the result of hadamard product, a list with [(news_id,embedding)]
        '''
        ids = [item[0] for item in text_embedding]
        text_emb = [item[1] for item in text_embedding]
        visual_emb = [item[1] for item in visual_embedding]
        fusion_emb = []
        for i in range(len(text_emb)):
            fusion_emb.append(torch.mul(text_emb[i],visual_emb[i]))
        return dict(zip(ids,fusion_emb))
    
    def _pointwise_add(self,text_embedding, visual_embedding):
        '''
        Pointwise Add.
        require the two embeddings with same dimension.

        parameter:
            text_embedding  : the result of text_model.encode(), a list with [(news_id,embedding)]
            visual_embedding: the result of visual_embedding.encode, a list with [(news_id,embedding)]
        return  :
            fusion_embedding: the result of hadamard product, a list with [(news_id,embedding)]
        '''
        ids = [item[0] for item in text_embedding]
        text_emb = [item[1] for item in text_embedding]
        visual_emb = [item[1] for item in visual_embedding]
        fusion_emb = []
        for i in range(len(text_emb)):
            fusion_emb.append(text_emb[i]+visual_emb[i])
        return dict(zip(ids,fusion_emb))
    
    def _concatenate(self,text_embedding, visual_embedding):
        '''
        concatenate. the dimension of embedding to return = text + visual.
        not require the same dimension.

        parameter:
            text_embedding  : the result of text_model.encode(), a list with [(news_id,embedding)]
            visual_embedding: the result of visual_embedding.encode, a list with [(news_id,embedding)]
        return  :
            fusion_embedding: the result of hadamard product, a list with [(news_id,embedding)]
        '''
        ids = [item[0] for item in text_embedding]
        text_emb = [item[1] for item in text_embedding]
        visual_emb = [item[1] for item in visual_embedding]
        fusion_emb = []
        for i in range(len(text_emb)):
            fusion_emb.append(torch.cat(text_emb[i],visual_emb[i]))
        return dict(zip(ids,fusion_emb))
    




def test_news_emb():
    stage1_config_path = "config_files//stage1_configs.json"
    stage1_configs = Config()
    stage1_configs.stage1_read_from_json(stage1_config_path)
    stage1_dataloader = Dataloader(stage1_configs)
    Emb_Pipeline = News_Embedding_Pipeline(stage1_configs)
    Emb_Pipeline.save_encode(stage1_dataloader)
    
def test_build_user_batch():
    stage1_config_path = "config_files//stage1_configs.json"
    stage1_configs = Config()
    stage1_configs.stage1_read_from_json(stage1_config_path)
    stage1_dataloader = Dataloader(stage1_configs)
    stage1_dataloader.build_user_batch()

def test_load_user_batch():
    stage1_config_path = "config_files//stage1_configs.json"
    stage1_configs = Config()
    stage1_configs.stage1_read_from_json(stage1_config_path)
    stage1_dataloader = Dataloader(stage1_configs)
    user_batch = stage1_dataloader.load_user_batch()
    
    for i in range(3):
        batch_data = next(user_batch)
        #print(batch_data)
        for data in batch_data:
            print(data[0])
            break

if __name__ == "__main__":
    test_build_user_batch()