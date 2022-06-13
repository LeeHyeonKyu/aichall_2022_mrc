"""loss 정의
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss(loss_name: str, ignore_index=None):
    if loss_name == "crossentropy":
        return nn.CrossEntropyLoss()


def cal_loss(start_positions, end_positions, start_logits, end_logits):
    '''MRC Task에서 Loss를 계산하는 함수'''
    total_loss=None

    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        
    return total_loss