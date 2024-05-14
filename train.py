import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torch
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from model import Model
from min_ddpm import MinDDPM
from datasets import make_swiss_roll, make_circles, get_data_loader
import tqdm

if __name__ == "__main__":
    model_config = {
        "in_dim": 2,
        "out_dim": 2,
        "embed_dim": 128,
        "num_layers": 4,
        "residual": False,
    }
    model = Model(**model_config)
    total_params = sum([p.numel() for p in model.parameters()])
    print("Model initialized, total params = ", total_params)

    ddpm_config = {"beta1": 1e-3, "beta2": 0.02, "T": 1000, "schedule": "linear"}
    ddpm = MinDDPM(model=model, **ddpm_config)

    train_config = {"lr": 1e-4, "batch_size": 256, "num_epochs": 10000}

    X = make_swiss_roll(10000, 0.1, 0.15)
    # X = make_circles(10000, 0.1, 1)
    loader = get_data_loader(X, train_config["batch_size"])

    opt = torch.optim.Adam(ddpm.parameters(), lr=train_config["lr"])
    lr_schedule = LinearLR(opt)
    opt.zero_grad()

    for e in range(train_config["num_epochs"]):
        loss_ema = None
        pbar = tqdm.tqdm(enumerate(loader))

        for idx, x in pbar:
            opt.zero_grad()
            t = torch.randint(1, ddpm_config["T"] + 1, (x.shape[0],))
            loss = ddpm(x, t)
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_description(f"epoch {e} loss: {loss_ema:.4f}")

            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.)
            opt.step()

    exp_name = "swissroll"
    working_dir = os.path.join("static", exp_name)
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)

    # save model
    print("Saving model ...")
    torch.save(ddpm.state_dict(), os.path.join(working_dir, "model.pth"))
    print("Done saving model!")

    # perform sampling
    x = torch.ones((1000, 2))
    x, hist = ddpm.sample(x)

    # save images
    print("Saving images ...")
    # create gif of images evolving over time, based on Xt_history
    def animate_diff(i, Xt_history):
        plt.cla()
        plots = sns.scatterplot(
            x=Xt_history[i][:, 0], y=Xt_history[i][:, 1], s=3, alpha=0.9
        )
        plots.set_xlim(-15, 15)
        plots.set_ylim(-15, 15)
        return plots

    fig = plt.figure(figsize=(5, 5))
    plt.clf()

    ani = FuncAnimation(
        fig,
        animate_diff,
        fargs=[hist],
        interval=200,
        blit=False,
        repeat=True,
        frames=len(hist),
    )
    ani.save(
        os.path.join(working_dir, "animation.gif"),
        dpi=100,
        writer=PillowWriter(fps=5),
    )
    print("Done saving images!")
