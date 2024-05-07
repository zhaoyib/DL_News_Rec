'''
@File      :   evaluator.py
@Time      :   2024/04/14 01:33:49
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   create evaluator.py
'''

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

class Evaluator(object):
    def __init__(self) -> None:
        pass
        
    def cor_measure(self,label,pred):
        pred = torch.argmax(pred,-1)
        count = 0
        label = label.cuda("cuda:0")
        if torch.eq(pred,label):
            count = count + 1
        return count

    def measures(self,y_true,y_rank):
        '''
        y_true = [1,0,1,0,0] type
        y_rank is [2,5,1,3,4] type, means the score of third place is highest. second place is lowest.
        '''
        y_score = []
        for rank in y_rank:
            y_score.append(1./rank)
            
        auc = roc_auc_score(y_true,y_score)
        mrr = self._mrr_score(y_true,y_score)
        ndcg5 = self._ndcg_score(y_true,y_score,5)
        ndcg10 = self._ndcg_score(y_true,y_score,10)
        return auc, mrr, ndcg5, ndcg10

    def MIND_measure(self,label,pred):
        raise NotImplementedError
    
    def _cor_f1(self,label,pred):
        pred = torch.argmax(pred,-1)
        count = 0
        label = label.cuda("cuda:0")
        if torch.eq(pred,label):
            count = count + 1
        return count

    def _dcg_score(self,y_true, y_score, k=10):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gains = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gains / discounts)

    def _ndcg_score(self,y_true, y_score, k=10):
        best = self._dcg_score(y_true, y_true, k)
        actual = self._dcg_score(y_true, y_score, k)
        return actual / best

    def _mrr_score(self,y_true, y_score):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order)
        rr_score = y_true / (np.arange(len(y_true)) + 1)
        return np.sum(rr_score) / np.sum(y_true)

    






