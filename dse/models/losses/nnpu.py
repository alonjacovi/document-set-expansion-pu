# from typing import Dict, Any, Optional, List
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# @loss.register("nnpu") # add with moving positive_class to forward
class NonNegativePULoss(nn.Module):
    def __init__(self, prior, positive_class=0, loss=None, gamma=1, beta=0, nnpu=True):
        super(NonNegativePULoss, self).__init__()
        self.prior = prior
        self.gamma = gamma
        self.beta = beta
        self.loss = loss or (lambda x: torch.sigmoid(-x))
        self.nnPU = nnpu
        self.positive = positive_class
        self.unlabeled = 1 - positive_class

    def forward(self, x, t):
        t = t[:, None]
        positive, unlabeled = (t == self.positive).float(), (t == self.unlabeled).float()
        n_positive, n_unlabeled = max(1., positive.sum().item()), max(1., unlabeled.sum().item())

        y_positive = self.loss(x)  # per sample positive risk
        y_unlabeled = self.loss(-x)  # per sample negative risk

        positive_risk = torch.sum(self.prior * positive / n_positive * y_positive)
        negative_risk = torch.sum((unlabeled / n_unlabeled - self.prior * positive / n_positive) * y_unlabeled)

        if self.nnPU:
            if negative_risk.item() < -self.beta:
                objective = (positive_risk
                             - self.beta + self.gamma * negative_risk).detach() - self.gamma * negative_risk
            else:
                objective = positive_risk + negative_risk
        else:
            objective = positive_risk + negative_risk

        return objective
