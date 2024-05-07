import os
from logging import getLogger

import torch
import json

def read_json_data(filename):
    '''
    load data from a json file
    '''
    f = open(filename, 'r', encoding="utf-8")
    return json.load(f)

def write_json_data(data, filename):
    """
    write data to a json file
    """
    with open(filename, 'w+', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    f.close()


class AbstractTrainer(object):
    """abstract trainer

    the base class of trainer class.
    
    example of instantiation:
        
        >>> trainer = AbstractTrainer(config, model, dataloader, evaluator)

        for training:
            
            >>> trainer.fit()
        
        for testing:
            
            >>> trainer.test()
        
        for parameter searching:

            >>> trainer.param_search()
    """

    def __init__(self, config, model, dataloader, evaluator):
        """
        Args:
            config (config): An instance object of Config, used to record parameter information.
            model (Model): An object of deep-learning model. 
            dataloader (Dataloader): dataloader object.
            evaluator (Evaluator): evaluator object.
        
        expected that config includes these parameters below:

        test_step (int): the epoch number of training after which conducts the evaluation on test.

        best_folds_accuracy (list|None): when running k-fold cross validation, this keeps the accuracy of folds that already run. 

        """
        super().__init__()
        self.config = config
        self.model = model
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.logger = getLogger()
        

        self.best_valid_cor_f1 = 0.
        self.best_test_cor_f1 = 0.
        self.start_epoch = 0
        self.epoch_i = 0
        self.output_result = []

        self._build_optimizer()

        if config.start_epoch != 0:
            self._load_checkpoint()

    def _save_checkpoint(self):
        raise NotImplementedError

    def _load_checkpoint(self):
        raise NotImplementedError

    def _save_model(self):
        state_dict = {"model": self.model.state_dict()}

        trained_model_path = self.config.trained_model_folder + self.config.model_name
        
        torch.save(state_dict, trained_model_path)
        
    def _load_model(self):

        load_dir = self.config.trained_model_folder

        model_file = os.path.join(load_dir, self.config.model_name)
        state_dict = torch.load(model_file, map_location=None)
        self.model.load_state_dict(state_dict["model"], strict=False)

    def _save_output(self):
        if not os.path.isabs(self.config.res_folder):
            output_dir = os.path.join(os.getcwd(),self.config.res_folder)
        else:
            output_dir = self.config.res_folder
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        save_dir = output_dir

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        output_file = os.path.join(save_dir, f'{self.config.model_name}_generation_result.json')
        write_json_data(self.output_result, output_file)

    def _build_optimizer(self):
        raise NotImplementedError

    def _train_batch(self):
        raise NotADirectoryError

    def _eval_batch(self):
        raise NotImplementedError

    def _train_epoch(self):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def evaluate(self, eval_set):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def param_search(self):
        raise NotImplementedError
