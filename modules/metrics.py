"""metric 정의
"""

import numpy
from sklearn.metrics import accuracy_score


def get_metric(metric_name):

    if metric_name == "accuracy":
        return accuracy_score
