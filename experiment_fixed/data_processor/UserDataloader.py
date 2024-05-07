'''
@File      :   UserDataloader.py
@Time      :   2024/04/11 17:20:22
@LastEdit  :   2024/04/13 20:33:49
@Author    :   YiboZhao 
@Version   :   1.0
@Site      :   https://github.com/zhaoyib
@RecentInfo:   TODO:my collect fn, to make the batch.
'''
from torch.utils.data import DataLoader, Dataset
import torch
from data_processor.BaseDataloader import BaseDataLoader
from utils.logger import logger_wrapper

class UserDataLoader(BaseDataLoader):
    def __init__(self, dataset, configs, **kwargs):
        batch_size, logger_path = self.parse_configs(configs)
        self.device = configs.device
        self.logger = logger_wrapper("UserDataLoader",path = logger_path)
        super().__init__(dataset, batch_size=batch_size, shuffle=True,num_workers = 0,collate_fn=self.build_batch_fn, **kwargs)

    def parse_configs(self,configs):
        batch_size = configs.batch_size
        logger_path = configs.logger_path
        return batch_size,logger_path

    def build_batch_fn(self,batch):
        '''
        format input of the batch:
            len(batch) = batchsize
            user = batch[i] is a user's data, including: 
            "id" : userid
            "seq": the embedded histories clicked news, a list of same shape tensor.
            "impression" : the new coming news. only one passage. a tensor with same shape with seq[0]
            "target" : if the user clicked the impression.
        
        format output of the batch:
            dimension 1: batchsize,
            dimension 2: max sequence length in the batch
            dimension 3: 768, the embedding dimension of the pretrained model.

        format output of the mask:
            a tensor, with the shape = [batchsize, max sequence length].

        format output of the target:
            a tensor, with the length = max sequence length.
        '''
        userid, output_batch, target  = self._build_sequence(batch)
        output_batch, src_mask = self._padding(output_batch)
        
        return {"userid":userid, "batch": output_batch, "mask":src_mask, "target":target}
    
    def _build_sequence(self,batch):
        '''
        concat the tensors, from [clicked_histories] and [impressions]
        to [clicked_histories | impression].
        return as a tensor at CUDA:0
        '''
        res = []
        targets = []
        userid = []
        for user in batch:
            userid.append(user["id"])
            clicked_histories = user["seq"]
            impression = user["impression"]
            sequence = clicked_histories + [impression]
            tensor_sequence = torch.stack(sequence)
            res.append(tensor_sequence)
            targets.append(user["target"])
        return userid, res, torch.tensor(targets)

    def _padding(self,batch):
        '''
        when the sequences built, they will have different length.
        it is used to pad the sequences to the longest in the batch.
        return the padded batch and the max length
        '''
        max_length = 50
        padded_sequences = []
        #print(len(batch))
        for seq in batch:
            padding_length = max_length - len(seq)
            if padding_length >= 0:
                padding_mask = torch.cat([torch.ones(len(seq)), torch.zeros(padding_length)])
                # 创建一个零张量作为填充部分
                padding_tensor = torch.zeros(padding_length, seq.size(1))
                # 将填充部分与原始 sequence 拼接起来
                padded_seq = torch.cat((seq, padding_tensor.to(self.device[0])), dim=0)
                # 将填充后的 sequence 添加到列表中
                padded_sequences.append(padded_seq)
            else:
                padding_mask = torch.ones(max_length)
                padded_seq = seq[-50:,:]
                padded_sequences.append(padded_seq)
        padded_tensor = torch.stack(padded_sequences).to(self.device[0])
        #print(padded_tensor.shape)
        return padded_tensor, padding_mask