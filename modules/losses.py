"""loss 정의
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss(loss_name: str):

    if loss_name == "crossentropy":

        return F.cross_entropy
