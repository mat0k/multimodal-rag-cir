import torch


def make_normalized(features: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(features, p=2, dim=-1)
