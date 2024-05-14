from torch import nn
import torch
import numpy as np


class Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, residual: bool = False) -> None:
        super(Block, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.residual = residual
        if self.residual:
            self.fc2 = nn.Linear(out_dim, out_dim)
            self.act2 = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        if self.residual:
            skip = x
            x = self.fc2(x)
            x = self.act2(x)
            x = x + skip
        return x


class SinusoidalPE(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super(SinusoidalPE, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> None:
        half_dims = self.embed_dim // 2
        embeddings = np.log(10000) / (half_dims - 1)
        embeddings = torch.exp(
            -embeddings * torch.arange(half_dims, device=t.device)
        )  # got 1/(10000^(k/d))
        embeddings = t[:, None] * embeddings[None, :]  # got frequencies
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class Model(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, embed_dim: int, num_layers: int, residual: bool
    ) -> None:
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        self.num_layers = num_layers

        self.t_emb = SinusoidalPE(embed_dim)
        self.layers = nn.ModuleList()
        for i in range(num_layers - 1):
            # make sure input dimension = dim(x) + dim(t_emb)
            if i == 0:
                fc = Block(in_dim + embed_dim, embed_dim)
            else:
                fc = Block(embed_dim + embed_dim, embed_dim)
            self.layers.append(fc)
        # add final fc without activation
        self.layers.append(nn.Linear(embed_dim + embed_dim, out_dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(
            -1,
        )
        t_emb = self.t_emb(t)
        # print("t_emb", t_emb.shape)
        for idx, fc in enumerate(self.layers):
            # print(f"fc{idx}")
            # print("     in ", x.shape, " and ", t_emb.shape)
            x = fc(torch.cat([x, t_emb], dim=1))
            # print("     out ", x.shape)

        return x


if __name__ == "__main__":
    model = Model(2, 2, 128, 3, False)
    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    x = torch.from_numpy(np.ones((2, 2)))
    t = torch.from_numpy(np.ones((2, 1)))
    x = x.float()
    t = t.float()
    model(x, t)
