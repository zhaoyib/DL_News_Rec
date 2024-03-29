'''
@File      :   data_loader.py
@Time      :   2024/03/19 15:29:15
@LastEdit  :   2024/03/21 16:50:15
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   updating user batch now, build first, then load.
'''

import cv2
import csv
import pickle as pkl
import os
import random
from typing import Union,List
from tqdm import tqdm
#from test_load_configs import Config # for module test only
from utils.load_configs import Config

class Dataloader():
    def __init__(self, configs:Config) -> None:
        self.mode = configs.mode
        assert self.mode.lower() in ["train","dev","test","valid"], \
        "Input Correct Mode in: train, dev, test, valid."
        self.data_path = configs.data_path
        self.batch_size = configs.batch_size
        self.news_data = None
        self.random_seed = configs.seed
        self.emb_path = configs.emb_path

    def load_news_batch(self):
        if self.news_data is None:
            self._load_raw_news_text()
        total_files = len(self.news_data)
        for i in range(total_files // self.batch_size + 1):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, total_files)
            id_list = []
            title_list = []
            abs_list = []
            #cat_list = []
            #subcat_list = []
            for j in range(start_index, end_index):
                id_list.append(self.news_data[j][0])
                title_list.append(self.news_data[j][3])
                abs_list.append(self.news_data[j][4])
                #cat_list.append(self.news_data[j][1])
                #subcat_list.append(self.news_data[j][2])
            img_list = self._load_raw_news_img(id_list)
            yield list(zip(id_list,title_list,abs_list,img_list))
    
    def load_user_batch(self):
        raise NotImplementedError("Dataloader.py-load_user_batch is coding now.")

    def build_user_batch(self):
        path = os.path.join(self.data_path,self.mode,"behaviors.tsv")
        with open(path,"r",encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            setattr(self,"user_data",list(reader))
        items = os.listdir(self.emb_path)
        for user in tqdm(self.user_data,desc="Building user batch now.",total=len(self.user_data)):
            user_id = user[1]
            histories = user[3]
            for history in histories:
                pass
        # for i in range(len(items)):
        #     item_path = os.path.join(self.emb_path,items[i])
        #     with open(item_path,"rb") as f:
        #         embeddings = pkl.load(f)
        #     yield embeddings # return a generator.

    def _load_raw_news_text(self):
        '''
        load news data from the news.tsv, text part.

        parameters:
            None.
        Do:
            add list of data in news.tsv to self.news_data
        return:
            None.
        '''
        path = os.path.join(self.data_path,self.mode,"news.tsv")
        with open(path,"r",encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            self.news_data = list(reader)
    
    def _load_raw_news_img(self,ids:Union[list,str]):
        path = os.path.join(self.data_path,"IM-MIND")
        if isinstance(ids,str):
            ids = [ids]
        res = []
        for id in ids:
            item_path = os.path.join(path,id+".jpg")
            img = cv2.imread(item_path)
            crop = img[15:180,15:215] # no whiteside now.
            crop_resize = cv2.resize(crop,(224,224))
            res.append(crop_resize)
        return res


'''
    def _load_raw_news_img(self, id):
        path = os.path.join(self.data_path,"IM-MIND")
        items = os.listdir(path)
        
        #return with the order 1,2,3,4…… instead of 1，10……
        def sort_key(filename):
            return int(filename[1:-4])
        
        items = sorted(items, key=sort_key)
        for item in items:
            news_id = item.split('.')[0]
            item_path = os.path.join(path,item)
            img = cv2.imread(item_path)
            crop = img[15:180,15:215]#now it is the img with no whiteside.
            self.imgs_data.append(crop)
'''

if __name__ == "__main__":
    path = "config_files//foo-config.json"
    test_config = Config()
    test_config.read_from_json(path)
    foo_dataset = Dataloader(test_config)
    one_batch = foo_dataset.load_news_batch()
    for tuple in one_batch:
        print(tuple[3])
        break
    #print(one_batch[0])