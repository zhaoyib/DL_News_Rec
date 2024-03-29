'''
@File      :   load_configs.py
@Time      :   2024/03/21 10:56:02
@LastEdit  :   2024/03/29 11:17:32
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   split it to stage1_read_from_json
               and stage2_read_from_json to adapt
               the requirement. Using setattr plz!
'''
import json
import torch
from typing import List,Dict,Tuple,Type,Union
from datetime import datetime

class Config():
    '''
    Using stage1_read_from_json to load the configs for News_Embedding.
    Using stage2_read_from_json to load the configs for train/test recommendation system
    detail info @{stage1_read_from_json,stage2_read_from_json}
    '''
    def __init__(self) -> None:
        pass

    def stage1_read_from_json(self, path:str):
        '''
        No finetuning. Using static pretrained model to emb.

        It will load the parameters including:
            mode = test, data_path, batch_size = 32
            model_set, emb_path, fusion_methods = HP
            device, logger_path, batch_size, seed
        '''
        with open(path,'r') as f:
            configs:dict = json.load(f)
        setattr(self,"mode",configs.get("mode","test"))
        assert self.mode.lower() in ["train","dev","test","valid","inference"],\
        "Input Correct Mode in: train, dev, test, valid, inference."
        #wont finetuning.
        self._load_emb_configs(configs)

        self._load_generic_config(configs)

    def stage2_read_from_json(self, path:str):
        '''
        Two settings, train or Inference.

        if train, it will load the parameters including:
            mode, device, logger_path, batch_size,seed
            start_epoch, end_epoch, model_path, model_set
            valid_freq, optimizer_set, ckp_path, ckp_epoch
        else, load the parameters including:
            mode, device, logger_path, batch_size,seed
            model_path, res_path
        '''
        with open(path,'r') as f:
            configs:dict = json.load(f)
        # if the mode hasn't been set, test as default.
        setattr(self,"mode",configs.get("mode",""))
        assert self.mode.lower() in ["train","dev","test","valid","inference"],\
        "Input Correct Mode in: train, dev, test, valid, inference."

        if self.mode.lower() == "train":
            self._load_train_configs(configs)
        else:
            self._load_inference_configs(configs)
        
        self._load_generic_config(configs)

    def _load_generic_config(self,configs:dict):
        '''
        load and check generic configs, including following parameters:
            device      : detact supporting device automatically.
            logger_path : where to write the log. str, default logger_files//log_mm-dd.txt
            batch_size  : how many samples train at a time, default 32
            seed        : random seed, default 1
        '''
        setattr(self,"emb_path",configs.get("emb_path",""))
        setattr(self,"device",configs.get("device",self.detact_device()))
        date = datetime.now().date().strftime("%m-%d")
        setattr(self,"logger_path",configs.get("logger_path",f"logger_files//log_{date}.txt"))
        setattr(self,"batch_size",configs.get("batch_size",32))
        setattr(self,"seed",configs.get("seed",1))
        setattr(self,"data_path",configs.get("data_path",""))
        assert self.data_path,\
        "Please input a valid data_path."

    def _load_train_configs(self,configs:dict):
        '''
        load and check configs for train including following parameters:
            start_epoch   : default as 0, if not 0, it will resume from an existing model.
            end_epoch     : Mandatory Parameter. greater than start epoch.
            model_path    : if start_epoch != 0, then load check_point from model_path
            model_set     : if start_epoch == 0, then load model_set to initialize the model.
            valid_freq    : how many epoches per valid. default 5.
            optimizer_set : set about optimizer. a dict.
            ckp_path      : the path to save the check point.
            ckp_epoch     : how many epoches per saving check point.
        '''
        setattr(self,"start_epoch",configs.get("start_epoch",0))
        setattr(self,"end_epoch",configs.get("end_epoch"))
        assert self.end_epoch > self.start_epoch,\
        "Please Check the start_epoch and end_epoch,\
            start_epoch should be less than end_epoch."
        if self.start_epoch != 0:
            setattr(self,"model_path", configs.get("model_path",""))
            assert self.model_path,\
            "Please add model_path to config.json\
                to resume, or set the start_epoch as 0"
        else:
            setattr(self,"model_set",configs.get("model_set",{}))
            assert self.model_set,\
            "Please add model_set to config.json\
                to initialize the model, or add model_path\
                    and start_epoch != to resume."
    
        setattr(self,"valid_freq",configs.get("valid_freq",5))
        setattr(self,"optimizer_set",configs.get("optimizer_set",{}))
        date = datetime.now().date().strftime("%m-%d")
        setattr(self,"ckp_path",configs.get("ckp_path",f"checkpoints//checkpoint_{date}.pth"))
        setattr(self,"ckp_epoch",configs.get("ckp_epoch",5))

    def _load_inference_configs(self,configs):
        '''
        load and check configs for inference including following parameters:
            model_path  : str of the model used for inference.
            res_path    : str of the path to store the result.
        '''
        setattr(self,"model_set",configs.get("model_set",""))
        assert self.model_set,\
        "Please add model_path to config.json to load model"
        setattr(self,"res_path",configs.get("res_path",""))

        

    def _load_emb_configs(self,configs):
        setattr(self,"model_set",configs.get("model_set",{}))
        assert self.model_set,\
        "Please input the model_set as a dict with elements"
        setattr(self,"visual_model_name",self.model_set.\
                        get("visual_model_name",
                        "facebook/dinov2-base"))
        setattr(self,"text_model_name",self.model_set.\
                        get("text_model_name",
            "maidalun1020/bce-embedding-base_v1"))
        setattr(self,"access_token",self.model_set.get("access_token",
                    "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"))
        
        setattr(self,"fusion_methods",configs.get("fusion_methods",
                                          ["Hadamard Product",
                                            "Pointwise Add",
                                            "Concatenate",
                                            "Cross Attention",
                                            "HP","PA","C","CA"]))
        setattr(self,"fusion_method",configs.get("fusion_method","HP"))
        #checke the format of fusion method.
        assert isinstance(self.fusion_method,Union[str,int]),\
        f"Please input the fusion_method with correct format,\
            expect str or int, got {type(self.fusion_method)}"
        if isinstance(self.fusion_method,str):
            assert self.fusion_method in self.fusion_methods,\
            f"NotImplementedError, fusion_method '{self.fusion_method}' not in\
                the implemented methods, please check the code.\
                Add your method or choose the implemented method"
        if isinstance(self.fusion_method,int):
            assert self.fusion_method < len(self.fusion_methods),\
            f"List Index Overflow, fusion_methods includes\
            {len(self.fusion_methods)} methods, you should\
            input an int in range [0,{len(self.fusion_methods)})\
            not {self.fusion_method}."
            self.fusion_method = self.fusion_methods[self.fusion_method]

    #not used.
    def _load_finetune_configs(self,configs):
        setattr(self,"model_set",configs.get("model_set",{}))
        assert self.model_set,\
        "Please input the model_set as a dict with elements"
        setattr(self,"visual_model_name",self.model_set.\
                        get("visual_model_name",
                        "facebook/dinov2-base"))
        setattr(self,"text_model_name",self.model_set.\
                        get("text_model_name",
            "maidalun1020/bce-embedding-base_v1"))
        setattr(self,"access_token", self.model_set.get("access_token",
                    "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"))

    @staticmethod
    def detact_device():
        num_gpus = torch.cuda.device_count()
        device = "cuda" if num_gpus > 0 else "cpu"
        return (device,num_gpus)