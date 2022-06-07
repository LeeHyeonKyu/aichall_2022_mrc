"""metric 정의
"""

from sklearn.metrics import accuracy_score
import numpy

def get_metric(metric_name):
        
    if metric_name == 'accuracy':
        return accuracy_score