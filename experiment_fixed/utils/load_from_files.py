'''
@File      :   load_from_files.py
@Time      :   2024/04/11 10:13:18
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   load_from_files, include the functions:
                load_tsv_data, process_news_data, load_img_data
                process_user_data, combine_news_img, combine_user_embedding
'''
# before test this part, plz change the relative import to abs import.

import csv
import pandas as pd
import os
import cv2
import pickle as pkl
from tqdm import tqdm
from utils.logger import logger_wrapper
from utils.load_configs import Config

def load_tsv_data(path):
    '''
    load tsv data from path.
    path should be a str, direct to the file to load.
    depends on the path, it will include two type:
    behaviours:
        the res will be a list, each element is:
        [num, userid, time, clicked, impression]
        impression is newsid-0/1, 0 is not click, 1 is clicked
    news:
        res will be a list, each element is :
        [newsid, tag, subtag, title, abstract, entities, relations]
    '''
    with open(path,"r",encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        res = list(reader)
        #print(len(res))
        #print(res[0])
    return res

def process_news_data(data):
    '''
    input the raw data loaded by load_tsv_data function.
    output the processed data, a list with tuple inside,
    [newsid, title, abstract]
    '''
    res = [[tup[0],tup[3],tup[4]] for tup in data]
    return res

def load_img_data(folder,idx):
    '''
    folder contains all of the imgs.
    idx means the idx of the news.
    '''

    item_path = os.path.join(folder,idx+".jpg")
    img = cv2.imread(item_path)
    res = img[15:180,15:215] # no whiteside now.
    #res = cv2.resize(crop,(224,224))
    return res 

def process_user_data(data):
    '''
    data should be loaded by load_tsv_data function.
    output the processed data, a list with tuple inside.
    [userid, clicked list, impression list]
    '''
    res = [[tup[1], tup[3].split(" "),tup[4].split(" ")] for tup in data]
    return res

def build_projector():
    '''
    a projector dict, key is the id in MIND Small
    value is the id in MIND.
    '''
    def read_csv_columns(file_path, key_col, value_col):
        data = pd.read_csv(file_path)
        csv_dict = dict(zip(data[key_col], data[value_col]))
        return csv_dict
        
    csv_file = "autodl-tmp/projector.csv"  
    first_col = "nid"  
    last_col = "IM-MIND-INDEX"    
    csv_dict = read_csv_columns(csv_file, first_col, last_col)
    return csv_dict

def combine_news_img(news_data,folder):
    '''
    news_data is the processed data of news
    folder is the folder of imgines
    '''
    projector = build_projector()
    for tup in tqdm(news_data,desc = "get images of news now."):
        small_id = tup[0]
        img = load_img_data(folder,projector[tup[0]])
        tup.append(img)
    return news_data

def combine_user_embedding(user_data, config):
    '''
    user_data is processed data of users,
    news_embedding is a dict, key is the news id, value is the embedding.
    '''
    emb_path = config.emb_path
    user_path = config.user_path
    with open(emb_path,'rb') as f:
        news_embedding = pkl.load(f)
    res = []
    for user in tqdm(user_data,desc = "combining user with news embeddings."):
        embeddings = []
        if len(user[1]) > 0 and user[1][0]:
            for news in user[1]:
                embeddings.append(news_embedding.get(news))

        for news in user[2]:
            target = int(news.split("-")[1])
            i_embedding = news_embedding.get(news.split("-")[0])
            res.append({"id":user[0],"seq":embeddings, "impression":i_embedding, "target":target})
        
    with open(user_path, 'wb') as f:
        pkl.dump(res,f)
    return res

def combine_user_embeddings_test(user_data, config):
    '''
    user_data is processed data of users,
    news_embedding is a dict, key is the news id, value is the embedding.
    '''
    emb_path = config.emb_path
    user_path = config.user_path
    with open(emb_path,'rb') as f:
        news_embedding = pkl.load(f)
    res = []
    if config.mode != "test":
        for user in tqdm(user_data,desc = "combining user with news embeddings."):
            embeddings = []
            if len(user[1]) > 0 and user[1][0]:
                for news in user[1]:
                    embeddings.append(news_embedding.get(news))
            
            targets = []
            i_embeddings = []
            for news in user[2]:
                target = int(news.split("-")[1])
                i_embedding = news_embedding.get(news.split("-")[0])
                targets.append(target)
                i_embeddings.append(i_embedding)
            res.append({"id":user[0],"seq":embeddings, "impression":i_embeddings, "target":targets})
    else:
        for user in tqdm(user_data,desc = "combining user with news embeddings."):
            embeddings = []
            if len(user[1]) > 0 and user[1][0]:
                for news in user[1]:
                    embeddings.append(news_embedding.get(news))
            
            targets = []
            i_embeddings = []
            for news in user[2]:
                i_embedding = news_embedding.get(news.split("-")[0])
                i_embeddings.append(i_embedding)
            res.append({"id":user[0],"seq":embeddings, "impression":i_embeddings, "target":targets})

    with open(user_path, 'wb') as f:
        pkl.dump(res,f)
    return res

def save_as_pkl(data,path):
    with open(path,'wb') as f:
        pkl.dump(data,f)

def extract_path_from_config(configs):
    '''
    configs.data_path
    configs.mode
    '''
    #print(configs.data_path)
    img_folder = os.path.join(configs.data_path,"IM-MIND")
    tsv_folder = os.path.join(configs.data_path,configs.mode)
    return tsv_folder,img_folder

def Get_news_data(configs):
    '''
    combination of the above method
    return a list of news data, the elements are:
    [newsid, title, abstract, img]
    '''
    tsv_folder, img_folder = extract_path_from_config(configs)
    #print(img_folder)
    news_path = os.path.join(tsv_folder,"news.tsv")
    news_data = load_tsv_data(news_path)
    news_data = process_news_data(news_data)
    news_data = combine_news_img(news_data, img_folder)
    #print(len(news_data))
    return news_data

def Get_user_data(configs):
    '''
    To keep the code same as Get_news_data
    create it.
    '''
    tsv_folder, _ = extract_path_from_config(configs)
    user_path = os.path.join(tsv_folder, "behaviors.tsv")
    user_data = load_tsv_data(user_path)
    user_data = process_user_data(user_data)
    return user_data

def load_embedded_user(configs):
    '''
    load the embedded user data.
    '''
    user_path = configs.user_path
    with open(user_path,'rb') as f:
        res = pkl.load(f)
    return res

def load_embedded_news(configs):
    '''
    '''
    news_path = configs.emb_path
    with open(news_path, 'rb') as f:
        res = pkl.load(f)
    return res