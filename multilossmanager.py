import torch
import numpy as np

import csv

class MultiLossManager():
    def __init__(self, batch_size, num_losses, num_for_rebalance, update_once=False):
        self.weight_initialized = False
        self.num_losses = num_losses
        self.update_once = update_once

        self.loss_weights = np.zeros(num_losses)
        self.train_losses = np.zeros((num_for_rebalance + batch_size, num_losses))

        self.initialize_weights()
        self.cur_ptr = 0

    def initialize_weights(self):
        self.loss_weights[:] = 1 / self.num_losses

    def get_total_loss(self, losses, current_batch_size, update=True, weights_list=None):
        """Compute the weighted loss summation
        """
        # can set weights manually
        if weights_list:
            self.loss_weights = weights_list

        for index in range(current_batch_size):
            if index == 0:
                loss = 0
            
            loss_1batch = 0
            for index_loss in range(self.num_losses):
                if self.loss_weights[index_loss] != 0:
                    loss_1batch += self.loss_weights[index_loss] * losses[index_loss][index]
            
            loss += loss_1batch / current_batch_size

            if update:
                for index_loss in range(self.num_losses):
                    self.train_losses[self.cur_ptr+index, index_loss] = losses[index_loss][index]

        if update:
            self.cur_ptr += current_batch_size

        return loss, self.cur_ptr

    def rebalancing(self, current_lambda, epoch, logfile=None):
        """Auto rebalancing loss weights according to 'Multi-loss Rebalancing Algorithm for Monocular Depth Estimation' ECCV 2020
        """
        temp_train_scores_mean = self.train_losses[:self.cur_ptr, :].mean(axis=0)
        total_loss = np.sum(temp_train_scores_mean * self.loss_weights)
        
        if not self.weight_initialized:
            for index_loss in range(self.num_losses):
                self.loss_weights[index_loss] = (total_loss * self.loss_weights[index_loss]) / temp_train_scores_mean[index_loss]

            # save previous record
            self.weight_initialized = True
            self.previous_total_loss = np.sum(temp_train_scores_mean * self.loss_weights)
            self.previous_loss = temp_train_scores_mean
        elif not self.update_once:
            previous_loss_weights = self.loss_weights
            if self.previous_total_loss > 0:
                for index_loss in range(self.num_losses):
                    adjust_term = 1 + current_lambda * ((total_loss/self.previous_total_loss) * (self.previous_loss[index_loss]/temp_train_scores_mean[index_loss]) - 1)
                    adjust_term = min(max(adjust_term, 1.0/2.0), 2.0/1.0)
                    self.loss_weights[index_loss] = previous_loss_weights[index_loss] * adjust_term

            # save previous record
            self.previous_total_loss = np.sum(temp_train_scores_mean * self.loss_weights)
            self.previous_loss = temp_train_scores_mean
        
        self.cur_ptr = 0

        if logfile:
            with open(logfile, 'a') as f:
                f.write(f'{epoch}\t{self.loss_weights[0]}\t{self.loss_weights[1]}\t{total_loss}\n')