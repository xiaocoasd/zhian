import torch
from torch.distributions import Distribution, Independent, Normal
def dist(*logits: torch.Tensor) -> Distribution:
    return Independent(Normal(*logits), 1)
