import numpy as np
import torch
import torch as t
from torch import nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(80 * 80, 80)
        self.linear3 = nn.Linear(80, 1)

    def forward(self, x):
        result = self.linear1(x)
        result = self.linear3(result)
        return torch.sigmoid(result)

    def prob_class_1(self, x):
        prob = self(t.from_numpy(x.astype(np.float32)))
        return prob.detach().numpy()