import numpy as np
from sklearn import datasets
import torch
from torch.utils.data import DataLoader


def make_swiss_roll(num_samples: int, noise: float, scale: float) -> torch.Tensor:
    X, t = datasets.make_swiss_roll(n_samples=num_samples, noise=noise)
    X = X[:, [0, 2]]
    X *= scale
    X = X.astype(float)
    return torch.from_numpy(X)

def make_circles(num_samples: int, noise: float, scale: float) -> torch.Tensor:
    X, t = datasets.make_circles(n_samples=num_samples, noise=0.01)
    X = X[:, [0, 1]]
    X *= scale
    X = X.astype(float)
    return torch.from_numpy(X)


def get_data_loader(X, batch_size, num_workers=5):
    loader = DataLoader(X, batch_size=batch_size, num_workers=num_workers)
    return loader
