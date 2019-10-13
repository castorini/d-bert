import math

import torch


class ConfusionMatrix(object):

    def __init__(self, dim):
        self.dim = dim
        self.matrix = torch.zeros(dim, dim)

    def ingest(self, scores, gt_labels):
        pred_labels = scores.max(1)[1]
        for pred_label, gt_label in zip(pred_labels.tolist(), gt_labels.tolist()):
            self.matrix[pred_label, gt_label] += 1

    @property
    def acc(self):
        return (self.matrix.diag().sum() / self.matrix.sum()).item()

    @property
    def metrics(self):
        return dict(acc=self.acc)

    def __repr__(self):
        return repr(self.matrix)


class BinaryConfusionMatrix(ConfusionMatrix):

    def __init__(self):
        super().__init__(2)

    @property
    def fp(self):
        return self.matrix[1, 0].item()

    @property
    def fn(self):
        return self.matrix[0, 1].item()

    @property
    def tp(self):
        return self.matrix[1, 1].item()

    @property
    def tn(self):
        return self.matrix[0, 0].item()

    @property
    def metrics(self):
        return dict(tp=self.tp, tn=self.fp, fp=self.fn, fn=self.tn, mcc=self.mcc, acc=self.acc)

    @property
    def mcc(self):
        num = self.tp * self.tn - self.fp * self.fn
        denom = math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * \
            (self.tn + self.fp) * (self.tn + self.fn))
        if denom == 0:
            return 0.
        return num / denom

