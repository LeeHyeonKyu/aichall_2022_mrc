"""loss 정의
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


def get_loss(loss_name: str, ignore_index=None):
    if loss_name == "crossentropy":
        return ce_loss
    elif loss_name == "joint":
        return joint_loss
    elif loss_name == "mix":
        return mix_loss
    elif loss_name == "multi":
        return multi_loss


def ce_loss(start_positions, end_positions, start_logits, end_logits, q_logit=None):
    """MRC Task에서 Loss를 계산하는 기본 함수"""
    total_loss = None

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


def joint_loss(start_positions, end_positions, start_logits, end_logits, q_logit=None):
    """start와 end의 joint loss를 계산하는 함수"""
    batch_size, length = start_logits.size()
    joint_logit = start_logits.unsqueeze(2) * end_logits.unsqueeze(1)
    joint_logit = torch.triu(joint_logit)
    joint_logit = joint_logit.reshape(batch_size, length * length)
    gt = start_positions * length + end_positions
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(joint_logit, gt)
    return loss


def mix_loss(start_positions, end_positions, start_logits, end_logits, q_logit=None):
    return (
        joint_loss(start_positions, end_positions, start_logits, end_logits)
        + ce_loss(start_positions, end_positions, start_logits, end_logits)
    ) / 2


def multi_loss(start_positions, end_positions, start_logits, end_logits, q_logit):
    loss_fn = nn.BCELoss()
    gt = ((start_positions + end_positions) != 0) * 1.0
    q_logit = q_logit.flatten()
    tot_loss = loss_fn(q_logit, gt) / 10
    tot_loss += ce_loss(start_positions, end_positions, start_logits, end_logits)
    return tot_loss
