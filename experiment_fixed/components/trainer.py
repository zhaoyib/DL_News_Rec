'''
@File      :   trainer.py
@Time      :   2024/04/14 00:34:03
@LastEdit  :   2024/04/14 02:38:25
@Author    :   Yibo Zhao 
@Version   :   1.0
@Github    :   https://github.com/zhaoyib
@RecentInfo:   test should be updated in order to 
                adapt the format of MIND dataset.
'''



import os
import time
from datetime import datetime
import math,json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
log_dir = "/root/tf-logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logger = SummaryWriter(log_dir=log_dir)

import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

from components.abstract_trainer import AbstractTrainer
from utils.logger import logger_wrapper

def time_since(s):
    """compute time

    Args:
        s (float): the amount of time in seconds.

    Returns:
        (str) : formatting time.
    """
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


class Trainer(AbstractTrainer):
    """supervised trainer, used to implement training, testing, parameter searching in supervised learning.
    
    example of instantiation:
        
        >>> trainer = SupervisedTrainer(config, model, dataloader, evaluator)

        for training:
            
            >>> trainer.fit()
        
        for testing:
            
            >>> trainer.test()
        
        for parameter searching:

            >>> trainer.param_search()
    """

    def __init__(self, config, model, train_dataloader, dev_dataloader, evaluator):
        """
        Args:
            config (config): An instance object of Config, used to record parameter information.
            model (Model): An object of deep-learning model. 
            dataloader (Dataloader): dataloader object.
            evaluator (Evaluator): evaluator object.
        """
        super(Trainer,self).__init__(config, model, train_dataloader, evaluator)
        self._build_optimizer()
        self.display_train_step = config.display_freq
        self.test_step = config.valid_freq
        self.class_num = 2
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.model = model.to(self.config.device[0])
        self.logger = logger_wrapper("Trainer",path = config.logger_path)

    
    def _build_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.model_set["lr"])

    def _save_checkpoint(self):
        check_pnt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "record_epoch": self.epoch_i,
        }
        model_file = self.config.ckp_folder + f"{self.config.model_name}_checkpoint_{self.epoch_i}" + ".pth"
        
        torch.save(check_pnt, model_file)
        date = datetime.now().date().strftime("%m-%d")
        np.save(self.config.ckp_folder + f"{self.config.model_name}_configs_{self.epoch_i}" + ".npy", self.config)

    def _load_checkpoint(self):
        model_file = self.config.ckp_folder + f"{self.config.model_name}_checkpoint_{self.config.start_epoch}" + ".pth"
        check_pnt = torch.load(model_file, map_location= None)#map_location is none
        # load parameter of model
        self.model.load_state_dict(check_pnt["model"])
        # load parameter of optimizer
        self.optimizer.load_state_dict(check_pnt["optimizer"])
        # other parameter
        self.start_epoch = check_pnt["record_epoch"]
        
    def _save_predit(self, predict_out):
        save_dir = self.config.res_folder
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        predict_file = os.path.join(save_dir, f'{self.model_name}_predicts.json')
        write_json_data(predict_out, predict_file)

    # def recover_num(self, rewrite_list, target_batch):
    #     new_rewrite_list = []
    #     source_unk_list = [w for w in target_batch if w not in self.dataloader.pretrained_tokenzier.vocab]
    #     unk_id = 0
    #     for w in rewrite_list:
    #         if w == self.dataloader.pretrained_tokenzier.unk_token and unk_id < len(source_unk_list):
    #             new_rewrite_list += [source_unk_list[unk_id]]
    #             unk_id += 1
    #         else:
    #             new_rewrite_list += [w]
    #     return new_rewrite_list

    def _train_batch(self, input_batch):
        batch_loss = self.model(input_batch)
        return batch_loss

    def _eval_batch(self, input_batch, is_display=False):

        gen_start = time.time()
        '''
        batch_loss = self.model(batch, self.dataloader)
        '''
        # print(input_batch["batch"].shape)
        # print(input_batch["mask"].shape)
        # print(input_batch["target"].shape)
        # test out is the prob of each class.
        targets = input_batch["target"]
        test_out = self.model.model_test(input_batch)
        batch_size = len(test_out)
        weight =torch.tensor([1,25]).to("cpu")
        loss = torch.nn.CrossEntropyLoss(weight=weight)
        
        evaluation_metrics = []
        predict_out = []
        losses = 0
        tag1_pred = []
        ground_truth = []
        for idx in range(batch_size):
            tag1_pred.append(test_out[idx][1].item())
            ground_truth.append(targets[idx].item())
            tag = targets[idx]
            # to one hot matrix.
            tag = F.one_hot(tag,2).float().to("cpu")
            # put predict to cpu.
            test = test_out[idx].to('cpu')
            # calculate the loss.
            loss_one = loss(test,tag)
            losses += loss_one
            pred_out = {'id': ' '.join(input_batch['userid'][idx]),
                        'tag': ' '.join(str(test_out[idx][0].item()))}
            predict_out.append(pred_out)

        
        sorted_pred = sorted(range(1, len(tag1_pred)+1), key=lambda k: tag1_pred[k-1], reverse=True)
        #sorted_truth = sorted(range(1, len(ground_truth)+1), key=lambda k: ground_truth[k-1], reverse=True)
        # the return of self.evaluator.MIND_measure is a tuple of evaluation metrics
        # including : auc, mrr, NDCG@10, NDCG@5, ordered.
        # 
        valid_auc, valid_mrr, valid_ndcg10, valid_ndcg5 = self.evaluator.measures(ground_truth, sorted_pred)

        return predict_out, valid_auc, valid_mrr, valid_ndcg10, valid_ndcg5, losses

    def _train_epoch(self):
        epoch_start_time = time.time()
        # build a dict with value in format : list.
        loss_total = defaultdict(list)
        # enable the dropout by using model.train() inherited from nn.Module.
        self.model.train()
        loss_in_total = 0
        for batch_idx, input_batch in enumerate(self.train_dataloader):
            # 0 -> 1.
            # batch = input_batch["batch"]
            # mask = input_batch["mask"]
            # tgt = input_batch["target"]
            self.global_batch += 1
            self.batch_idx = batch_idx + 1
            # stop update the parameters.
            self.optimizer.zero_grad()
            batch_loss = self._train_batch(input_batch)
            # calculate the total loss in the epoch.
            for k in batch_loss:
                if 'loss' in k:
                    # torch.tensor.item() is used to retrieve value from the single value tensor.
                    # if more than 1 value, use the tolist to retrieve value.
                    loss_total[k] += [batch_loss[k].item()]
            # backforward process.
            batch_loss["loss"].backward()
            loss_in_total += batch_loss["loss"]
            if self.global_batch % 500 == 0:
                print("global_step:{}, loss:{:.4}".format(self.global_batch,batch_loss["loss"]))
                logger.add_scalar("train_loss", loss_in_total / 500 ,global_step=self.global_batch)
                loss_in_total = 0
            # use the clip_grad_norm_（梯度裁剪） to avoid gradient explosion.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            # optimizer is the Adam
            self.optimizer.step()
            # if batch_idx > 100:
            #     break
            #if batch_idx %50 == 0:
            #    print("50 batches finished")
            #if batch_idx > 300:
             #break
            # if batch_idx > 300:
            #     break

        self.optimizer.zero_grad()
        self.model.zero_grad()
        epoch_time_cost = time_since(time.time() - epoch_start_time)
        return loss_total, epoch_time_cost

    def fit(self):
        """
        train model.
        """
        train_batch_size = self.config.batch_size
        epoch_nums = self.config.end_epoch
        self.global_batch = 0

        self._save_checkpoint()
        
        self.train_batch_nums = len(self.train_dataloader)
        
        self.logger.info("start training...")
        #self.test()
        for epo in tqdm(range(self.start_epoch, epoch_nums), desc='train process '):
            # 0 -> 1
            self.epoch_i = epo + 1
            # enable the dropout.
            self.model.train()
            # call self._train_epoch to get the loss in this epoch.
            loss_total, train_time_cost = self._train_epoch()

            # if it is time to show the result of the epoch:
            if epo % self.display_train_step == 0 or epo > epoch_nums - 5:
                # build the logging out put.
                logging_output = ""
                for l in loss_total:
                    # iterate all kind of loss calculated. only when using MIND-measure, it will be 3 metrics.
                    # when train or normal valid, only the cor_f1 as the metric.
                    logging_output += " " + l + " |"
                    logging_output += "[%2.3f]" %(np.sum(loss_total[l])*1./self.train_batch_nums)

                # info the condition of the training.
                self.logger.info("epoch [%3d] train time %s | " %(self.epoch_i, train_time_cost) + logging_output)
                # self.logger.info("epoch [%3d] avr loss [%2.8f] | train time %s" \
                #                  % (self.epoch_i, loss_total / self.train_batch_nums, train_time_cost))
            
            if epo >= 0 and (epo % self.test_step == 0) or (epo > epoch_nums - 5):
                torch.cuda.empty_cache()
                _, valid_auc,valid_mrr,valid_ndcg5,valid_ndcg10, valid_loss ,valid_total, valid_time_cost = self.evaluate(self.dev_dataloader)
                logger.add_scalar("valid_auc", valid_auc, global_step=epo)
                logger.add_scalar("valid_mrr", valid_mrr, global_step=epo)
                logger.add_scalar("valid_ndcg10", valid_ndcg10, global_step=epo)
                logger.add_scalar("valid_ndcg5", valid_ndcg5, global_step=epo)
                
                torch.cuda.empty_cache()
                self.logger.info(
                    "---------- valid total [%d] | valid auc [%2.3f] | valid mrr [%2.3f] | valid ndcg@5 [%2.3f] | valid ndcg@10 [%2.3f] | valid loss [%2.3f] | valid time %s" \
                    % (valid_total, valid_auc, valid_mrr, valid_ndcg5, valid_ndcg10, valid_loss, valid_time_cost))

                # self.test()
                
                torch.cuda.empty_cache()                
                self._save_checkpoint()
                print("checkpoint saved")
            
        self.logger.info('''training finished.''')

    def evaluate(self, eval_set):
        """evaluate model.

        Args:
            eval_set (DataLoader): a built DataLoader. dev_dataloader.

        Returns:
            tuple(float,float,int,str):
            detection F1, correction F1, count of evaluated datas, formatted time string of evaluation time.
        """
        # no parameters will be updated now.
        self.model.eval()
        self.model.zero_grad()
        # calculate the metrics.
        aucs = 0
        mrrs = 0
        ndcg10s = 0
        ndcg5s = 0
        eval_total = len(eval_set)
        loss = 0
        predict_out = []

        batch_nums = len(eval_set)
        
        test_start_time = time.time()
        
        for batch_idx, batch in enumerate(eval_set):

            pred_out, batch_auc, batch_mrr, batch_ndcg10, batch_ndcg5 , batch_loss = self._eval_batch(batch, is_display = False)
            # Process the metrics cor f1.
            aucs += batch_auc
            mrrs += batch_mrr
            ndcg10s += batch_ndcg10
            ndcg5s += batch_ndcg5
            loss += batch_loss
            # what is it? not important, just for keeping the same format of output.
            # when using evaluate in fit in line 254, the first out put received by _
            predict_out += pred_out
            # if batch_idx > 100:
            #     break

        test_time_cost = time_since(time.time() - test_start_time)
        # avg loss or what.
        return predict_out, aucs / eval_total, mrrs / eval_total, ndcg10s / eval_total, ndcg5s / eval_total, loss/eval_total ,eval_total, test_time_cost

    def test(self):
        """test model.
        """
        self.model.eval()
        predict_out = [["userid","rank"]]
        test_start_time = time.time()

        for batch in self.test_dataloader:
            pred_out = self.model.model_test(batch)
            userid = batch['userid']
            #print(pred_out)
            pred_out = pred_out.argmax(dim=-1)
            pred_out = pred_out.tolist()
            print("trainer.py line 336, pred_out",pred_out)
            predict_out.append([userid,pred_out])

        test_time_cost = time_since(time.time() - test_start_time)
        self.logger.info("test time %s" \
                         % (test_time_cost))
        self._save_output()
        self._save_predit(predict_out)


