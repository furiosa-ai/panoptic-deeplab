import torch
from torch.nn import functional as F

t = torch.tensor([], dtype=Float)

A = F.interpolate(
    t, size=(1024, 2048), mode="nearest-exact"
)[0]

B = 