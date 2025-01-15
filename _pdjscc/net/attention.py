import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SNRAttention(nn.Module):

    def __init__(self, C):
        super(SNRAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(C + 1, (C + 1) // 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear((C + 1) // 16, C)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, SNR):
        feature_pooling = self.avg_pool(x)
        [b, c, _, _] = feature_pooling.shape
        context_information = torch.cat((SNR, feature_pooling.reshape(b, c)), 1)
        scale_factor = self.sigmoid(self.fc2(self.relu1(self.fc1(context_information))))
        out = torch.mul(x, scale_factor.unsqueeze(2).unsqueeze(3))
        return out
