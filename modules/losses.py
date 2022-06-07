"""loss 정의
"""

from torch.nn import functional as F
import torch
import torch.nn as nn


def get_loss(loss_name: str):

    if loss_name == 'crossentropy':

        return F.cross_entropy
