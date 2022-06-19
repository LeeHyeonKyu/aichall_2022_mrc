"""Optimizer 정의
"""

import torch.optim as optim
import adamp


def get_optimizer(optimizer_name: str):

    if optimizer_name == "sgd":

        return optim.SGD

    elif optimizer_name == "adam":

        return optim.Adam

    elif optimizer_name == "adamw":

        return optim.AdamW

    elif optimizer_name == "adamp":

        return adamp.AdamP