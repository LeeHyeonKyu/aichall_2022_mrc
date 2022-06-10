from timm.scheduler import create_scheduler
from types import SimpleNamespace


def get_scheduler(scheduler_cfg, optimizer):
    return create_scheduler(optimizer=optimizer, args=SimpleNamespace(**scheduler_cfg))