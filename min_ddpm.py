import torch
from torch import nn
import numpy as np
import tqdm

def get_noise_schedule(
    beta1: float, beta2: float, T: int, schedule: str = "linear"
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta1, beta2, T + 1)


def precalculate_params(betas: torch.Tensor) -> torch.Tensor:
    alphas = 1 - betas
    alphas_cum = torch.cumprod(alphas, dim=0)
    sqrt_alphas = torch.sqrt(alphas)
    sqrt_one_minus_alphas_cum = torch.sqrt(1 - alphas_cum)
    return alphas, alphas_cum, sqrt_alphas, sqrt_one_minus_alphas_cum


class MinDDPM(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        beta1: float,
        beta2: float,
        T: int,
        schedule: str = "linear",
    ) -> None:
        super(MinDDPM, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.T = T
        self.schedule = schedule
        self.model = model
        self.loss = nn.MSELoss()

        self.betas = get_noise_schedule(beta1, beta2, T, schedule)
        (
            self.alphas,
            self.alphas_cum,
            self.sqrt_alphas,
            self.sqrt_one_minus_alphas_cum,
        ) = precalculate_params(self.betas)

    def forward_diffusion(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.randn_like(x)
        x_t = torch.sqrt(self.alphas_cum[t].view(-1,1)) * x + torch.sqrt(1 - self.alphas_cum[t].view(-1,1)) * z
        return x_t, z

    def reverse_diffusion(
        self, x: torch.Tensor, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        z = torch.randn_like(x)
        t_ = torch.tensor(t)
        t_ = t_.repeat(x.shape[0])
        e = self.model(x, t_ / self.T)

        if t <= 1:
            z = 0

        x_prev = (
            1
            / torch.sqrt(self.alphas[t])
            * (x - (1-self.alphas[t]) / torch.sqrt(1-self.alphas_cum[t]) * e)
            + torch.sqrt(self.betas[t]) * z
        )

        return x_prev, z

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x_t, z = self.forward_diffusion(x, t)
        z_pred = self.model(x_t, t / self.T)
        loss = self.loss(z, z_pred)
        return loss

    def sample(self, shape: torch.Tensor) -> tuple[np.array, list[np.array]]:
        x = torch.randn_like(shape)
        hist = [x.detach().cpu().numpy()]
        print("Start sampling ...")
        for t in tqdm.tqdm(reversed(range(1, self.T+1))):
            x, _ = self.reverse_diffusion(x, t)
            if t % 10 ==0 or t<20:
                hist.append(x.detach().cpu().numpy())
        print("Done sampling!")
        
        return x.detach().cpu().numpy(), hist