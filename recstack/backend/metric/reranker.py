import statistics as stats
import numpy as np
import torch
from sklearn import metrics as sk_metrics

from .metric import Metric
from ..registry import METRIC

@METRIC.register("reranker_metrics")
class BinaryClassification(Metric):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        self.metric_functions = {
            "auc": sk_metrics.roc_auc_score,
            "accuracy": lambda y, y_hat: sk_metrics.accuracy_score(y, y_hat > self.threshold),
        }
        self.reset()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update(self, y, y_hat):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.detach().cpu().numpy()

        for name, fn in self.metric_functions.items():
            self.scores[name].append(fn(y, self.sigmoid(y_hat)))

    def mean(self):
        return {m: stats.mean(v) for m, v in self.scores.items() if v}

    def stdev(self):
        return {m: stats.pstdev(v) for m, v in self.scores.items() if v}

    def reset(self):
        self.scores = {name: [] for name in self.metric_functions}

    def __call__(self, y, y_hat):
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(y_hat, torch.Tensor):
            y_hat = y_hat.detach().cpu().numpy()

        return {name: fn(y, self.sigmoid(y_hat)) for name, fn in self.metric_functions.items()}