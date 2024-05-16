import numpy as np
from sklearn import datasets
import torch
import pandas as pd
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

def make_dino_dataset(num_samples:int=8000)->torch.Tensor:
    df = pd.read_csv("static/DatasaurusDozen.tsv", sep="\t")
    df = df[df["dataset"] == "dino"]

    rng = np.random.default_rng(42)
    ix = rng.integers(0, len(df), num_samples)
    x = df["x"].iloc[ix].tolist()
    x = np.array(x) + rng.normal(size=len(x)) * 0.15
    y = df["y"].iloc[ix].tolist()
    y = np.array(y) + rng.normal(size=len(x)) * 0.15
    x = (x/54 - 1) * 4
    y = (y/48 - 1) * 4
    X = np.stack((x, y), axis=1)
    return X

def get_data_loader(X, batch_size, num_workers=5):
    loader = DataLoader(X, batch_size=batch_size, num_workers=num_workers)
    return loader
