#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#--------------------------------------------------#
###################################################################################################################
###################################################################################################################
# Imports
import random
#--------------------------------------------------#
import math
import numpy as np

from tqdm import tqdm
#--------------------------------------------------#
from typing import List, Union, Any, Optional
#--------------------------------------------------#
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

#--------------------------------------------------#
from ZX03_nn_args import TrainArgs


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Contents
# func:  param_count
# func:  param_count_all
# class: StandardScaler





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MN.   `7MF'`7MN.   `7MF'    `7MM"""Mq.                                                                                                     M      #
#    MMN.    M    MMN.    M        MM   `MM.                                                                                                    M      #
#    M YMb   M    M YMb   M        MM   ,M9 ,6"Yb. `7Mb,od8 ,6"Yb. `7MMpMMMb.pMMMb.  ,pP"Ybd                                                    M      #
#    M  `MN. M    M  `MN. M        MMmmdM9 8)   MM   MM' "'8)   MM   MM    MM    MM  8I   `"                                                `7M'M`MF'  #
#    M   `MM.M    M   `MM.M        MM       ,pm9MM   MM     ,pm9MM   MM    MM    MM  `YMMMa.                                                  VAM,V    #
#    M     YMM    M     YMM        MM      8M   MM   MM    8M   MM   MM    MM    MM  L.   I8                                                   VVV     #
#  .JML.    YM  .JML.    YM      .JMML.    `Moo9^Yo.JMML.  `Moo9^Yo.JMML  JMML  JMML.M9mmmP'                                                    V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def param_count(model: nn.Module) -> int:
    # Determines number of trainable parameters.
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def param_count_all(model: nn.Module) -> int:
    # Determines number of trainable parameters.
    return sum(param.numel() for param in model.parameters())


def initialize_weights(model: nn.Module) -> None:
    """
    Initializes the weights of a model in place.

    :param model: An PyTorch model.
    """
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   .M"""bgd   mm                              `7MM                         `7MM   .M"""bgd                   `7MM                              M      #
#  ,MI    "Y   MM                                MM                           MM  ,MI    "Y                     MM                              M      #
#  `MMb.     mmMMmm   ,6"Yb.  `7MMpMMMb.    ,M""bMM   ,6"Yb.  `7Mb,od8   ,M""bMM  `MMb.      ,p6"bo   ,6"Yb.    MM   .gP"Ya  `7Mb,od8           M      #
#    `YMMNq.   MM    8)   MM    MM    MM  ,AP    MM  8)   MM    MM' "' ,AP    MM    `YMMNq. 6M'  OO  8)   MM    MM  ,M'   Yb   MM' "'       `7M'M`MF'  #
#  .     `MM   MM     ,pm9MM    MM    MM  8MI    MM   ,pm9MM    MM     8MI    MM  .     `MM 8M        ,pm9MM    MM  8M""""""   MM             VAM,V    #
#  Mb     dM   MM    8M   MM    MM    MM  `Mb    MM  8M   MM    MM     `Mb    MM  Mb     dM YM.    , 8M   MM    MM  YM.    ,   MM              VVV     #
#  P"Ybmmd"    `Mbmo `Moo9^Yo..JMML  JMML. `Wbmd"MML.`Moo9^Yo..JMML.    `Wbmd"MML.P"Ybmmd"   YMbmd'  `Moo9^Yo..JMML. `Mbmmd' .JMML.             V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none.astype('float32')


def normalize_targets(y_data) -> StandardScaler:
    # For Future Use.
    """
    Normalizes the targets of the dataset using a :class:`~chemprop.data.StandardScaler`.

    The :class:`~chemprop.data.StandardScaler` subtracts the mean and divides by the standard deviation
    for each task independently.

    This should only be used for regression datasets.

    :return: A :class:`~chemprop.data.StandardScaler` fitted to the targets.
    """
    scaler = StandardScaler().fit(y_data)
    scaled_targets = scaler.transform(y_data).tolist()

    return scaled_targets, scaler




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM                .M"""bgd          `7MM                      `7MM              `7MM                                                       M      #
#    MM               ,MI    "Y            MM                        MM                MM                                                       M      #
#    MM  `7Mb,od8     `MMb.      ,p6"bo    MMpMMMb.   .gP"Ya    ,M""bMM  `7MM  `7MM    MM   .gP"Ya  `7Mb,od8                                    M      #
#    MM    MM' "'       `YMMNq. 6M'  OO    MM    MM  ,M'   Yb ,AP    MM    MM    MM    MM  ,M'   Yb   MM' "'                                `7M'M`MF'  #
#    MM    MM         .     `MM 8M         MM    MM  8M"""""" 8MI    MM    MM    MM    MM  8M""""""   MM                                      VAM,V    #
#    MM    MM         Mb     dM YM.    ,   MM    MM  YM.    , `Mb    MM    MM    MM    MM  YM.    ,   MM                                       VVV     #
#  .JMML..JMML.       P"Ybmmd"   YMbmd'  .JMML  JMML. `Mbmmd'  `Wbmd"MML.  `Mbod"YML..JMML. `Mbmmd' .JMML.                                      V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    params = [{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}]

    return Adam(params)

###################################################################################################################
###################################################################################################################

def build_lr_scheduler(
    optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None ) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=total_epochs or [args.epochs] * args.num_lrs,
        steps_per_epoch=args.train_data_size // args.batch_size,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )



###################################################################################################################
###################################################################################################################
class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where :code:`warmup_steps = warmup_epochs * steps_per_epoch`).
    Then the learning rate decreases exponentially from :code:`max_lr` to :code:`final_lr` over the
    course of the remaining :code:`total_steps - warmup_steps` (where :code:`total_steps =
    total_epochs * steps_per_epoch`). This is roughly based on the learning rate
    schedule from `Attention is All You Need <https://arxiv.org/abs/1706.03762>`_, section 5.3.
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after :code:`warmup_epochs`).
        :param final_lr: The final learning rate (achieved after :code:`total_epochs`).
        """
        if not (
            len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs)
            == len(init_lr) == len(max_lr) == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates! "
                f"got: len(optimizer.param_groups)= {len(optimizer.param_groups)}, "
                f"len(warmup_epochs)= {len(warmup_epochs)}, "
                f"len(total_epochs)= {len(total_epochs)}, "
                f"len(init_lr)= {len(init_lr)}, "
                f"len(max_lr)= {len(max_lr)}, "
                f"len(final_lr)= {len(final_lr)}"
            )

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """
        Gets a list of the current learning rates.

        :return: A list of the current learning rates.
        """
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
                             If None, :code:`current_step = self.current_step + 1`.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#        db               mm     db              db            db                         `7MM"""YMM                                            M      #
#       ;MM:              MM                                                                MM    `7                                            M      #
#      ,V^MM.    ,p6"bo mmMMmm `7MM `7M'   `MF'`7MM  ,pP"Ybd `7MM  ,pW"Wq.`7MMpMMMb.        MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo                M      #
#     ,M  `MM   6M'  OO   MM     MM   VA   ,V    MM  8I   `"   MM 6W'   `Wb MM    MM        MM""MM   MM    MM   MM    MM 6M'  OO            `7M'M`MF'  #
#     AbmmmqMA  8M        MM     MM    VA ,V     MM  `YMMMa.   MM 8M     M8 MM    MM        MM   Y   MM    MM   MM    MM 8M                   VAM,V    #
#    A'     VML YM.    ,  MM     MM     VVV      MM  L.   I8   MM YA.   ,A9 MM    MM        MM       MM    MM   MM    MM YM.    ,              VVV     #
#  .AMA.   .AMMA.YMbmd'   `Mbmo.JMML.    W     .JMML.M9mmmP' .JMML.`Ybmd9'.JMML  JMML.    .JMML.     `Mbod"YML.JMML  JMML.YMbmd'                V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    Supports:

    * :code:`ReLU`
    * :code:`LeakyReLU`
    * :code:`PReLU`
    * :code:`tanh`
    * :code:`SELU`
    * :code:`ELU`

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')







#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    .g8""8q.     mm   `7MM                           `7MM"""YMM                                                                                M      #
#  .dP'    `YM.   MM     MM                             MM    `7                                                                                M      #
#  dM'      `MM mmMMmm   MMpMMMb.  .gP"Ya `7Mb,od8      MM   d `7MM  `7MM `7MMpMMMb.  ,p6"bo  ,pP"Ybd                                           M      #
#  MM        MM   MM     MM    MM ,M'   Yb  MM' "'      MM""MM   MM    MM   MM    MM 6M'  OO  8I   `"                                       `7M'M`MF'  #
#  MM.      ,MP   MM     MM    MM 8M""""""  MM          MM   Y   MM    MM   MM    MM 8M       `YMMMa.                                         VAM,V    #
#  `Mb.    ,dP'   MM     MM    MM YM.    ,  MM          MM       MM    MM   MM    MM YM.    , L.   I8                                          VVV     #
#    `"bmmd"'     `Mbmo.JMML  JMML.`Mbmmd'.JMML.      .JMML.     `Mbod"YML.JMML  JMML.YMbmd'  M9mmmP'                                           V      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in :code:`index`.

    :param source: A tensor of shape :code:`(num_bonds, hidden_size)` containing message features.
    :param index: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds)` containing the atom or bond
                  indices to select from :code:`source`.
    :return: A tensor of shape :code:`(num_atoms/num_bonds, max_num_bonds, hidden_size)` containing the message
             features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target































