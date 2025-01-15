import numpy as np
import os, sys
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

print(np.hanning(5))
a = np.hanning(5)[1:-1]
print(a)
g = torch.Tensor(a[:, None] * a[None, :])
print(g)
g = g / torch.sum(g)

