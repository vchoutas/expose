
import math
import os
import json
import time
import logging
import random
import datetime

import torch
import numpy as np

from .trainer import Trainer


class HypTuning:

    def __init__(self, joints, epochs, baseline, dropout, multiplier=1, r_seed=1):
        """
        Initialize directories, load the data and parameters for the training
        """

        # Initialize Directories
        self.joints = joints
        self.baseline = baseline
        self.dropout = dropout
        self.num_epochs = epochs
        self.baseline = baseline
        self.r_seed = r_seed
        dir_out = os.path.join('data', 'models')
        dir_logs = os.path.join('data', 'logs')
        assert os.path.exists(dir_out), "Output directory not found"
        if not os.path.exists(dir_logs):
            os.makedirs(dir_logs)

        name_out = 'hyp-baseline-' if baseline else 'hyp-monoloco-'

        self.path_log = os.path.join(dir_logs, name_out)
        self.path_model = os.path.join(dir_out, name_out)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize grid of parameters
        random.seed(r_seed)
        np.random.seed(r_seed)
        self.sched_gamma_list = [0.8, 0.9, 1, 0.8, 0.9, 1] * multiplier
        random.shuffle(self.sched_gamma_list)
        self.sched_step = [10, 20, 30, 40, 50, 60] * multiplier
        random.shuffle(self.sched_step)
        self.bs_list = [64, 128, 256, 512, 1024, 2048] * multiplier
        random.shuffle(self.bs_list)
        self.hidden_list = [256, 256, 256, 256, 256, 256] * multiplier
        random.shuffle(self.hidden_list)
        self.n_stage_list = [3, 3, 3, 3, 3, 3] * multiplier
        random.shuffle(self.n_stage_list)
        # Learning rate
        aa = math.log(0.001, 10)
        bb = math.log(0.03, 10)
        log_lr_list = np.random.uniform(aa, bb, int(6 * multiplier)).tolist()
        self.lr_list = [10 ** xx for xx in log_lr_list]
        # plt.hist(self.lr_list, bins=50)
        # plt.show()

    def train(self):
        """Train multiple times using log-space random search"""

        best_acc_val = 20
        dic_best = {}
        dic_err_best = {}
        start = time.time()
        cnt = 0
        for idx, lr in enumerate(self.lr_list):
            bs = self.bs_list[idx]
            sched_gamma = self.sched_gamma_list[idx]
            sched_step = self.sched_step[idx]
            hidden_size = self.hidden_list[idx]
            n_stage = self.n_stage_list[idx]

            training = Trainer(joints=self.joints, epochs=self.num_epochs,
                               bs=bs, baseline=self.baseline, dropout=self.dropout, lr=lr, sched_step=sched_step,
                               sched_gamma=sched_gamma, hidden_size=hidden_size, n_stage=n_stage,
                               save=False, print_loss=False, r_seed=self.r_seed)

            best_epoch = training.train()
            dic_err, model = training.evaluate()
            acc_val = dic_err['val']['all']['mean']
            cnt += 1
            print("Combination number: {}".format(cnt))

            if acc_val < best_acc_val:
                dic_best['lr'] = lr
                dic_best['joints'] = self.joints
                dic_best['bs'] = bs
                dic_best['baseline'] = self.baseline
                dic_best['sched_gamma'] = sched_gamma
                dic_best['sched_step'] = sched_step
                dic_best['hidden_size'] = hidden_size
                dic_best['n_stage'] = n_stage
                dic_best['acc_val'] = dic_err['val']['all']['mean']
                dic_best['best_epoch'] = best_epoch
                dic_best['random_seed'] = self.r_seed
                # dic_best['acc_test'] = dic_err['test']['all']['mean']

                dic_err_best = dic_err
                best_acc_val = acc_val
                model_best = model

        # Save model and log
        now = datetime.datetime.now()
        now_time = now.strftime("%Y%m%d-%H%M")[2:]
        self.path_model = self.path_model + now_time + '.pkl'
        torch.save(model_best.state_dict(), self.path_model)
        with open(self.path_log + now_time, 'w') as f:
            json.dump(dic_best, f)
        end = time.time()
        print('\n\n\n')
        self.logger.info(" Tried {} combinations".format(cnt))
        self.logger.info(" Total time for hyperparameters search: {:.2f} minutes".format((end - start) / 60))
        self.logger.info(" Best hyperparameters are:")
        for key, value in dic_best.items():
            self.logger.info(" {}: {}".format(key, value))

        print()
        self.logger.info("Accuracy in each cluster:")

        for key in dic_err_best['val']:
            self.logger.info("Val: error in cluster {} = {} ".format(key, dic_err_best['val'][key]['mean']))
        self.logger.info("Final accuracy Val: {:.2f}".format(dic_best['acc_val']))
        self.logger.info("\nSaved the model: {}".format(self.path_model))
