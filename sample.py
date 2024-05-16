import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import torch
from model import Model
from min_ddpm import MinDDPM

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

    ddpm_config = {"beta1": 1e-4, "beta2": 0.02, "T": 1000, "schedule": "linear"}
    ddpm = MinDDPM(model=model, **ddpm_config)

    working_dir = "static/swissroll"
    model_name = "model.pth"
    ddpm.load_state_dict(torch.load(os.path.join(working_dir, model_name)))

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
        plots.set_xlim(-5, 5)
        plots.set_ylim(-5, 5)
        plots.set_title(f"time step {i*5}")
        return plots

    fig = plt.figure(figsize=(5, 5))
    plt.clf()

    ani = FuncAnimation(
        fig,
        animate_diff,
        fargs=[hist],
        interval=50,
        blit=False,
        repeat=False,
        frames=len(hist),
    )
    ani.save(
        os.path.join(working_dir, "animation.gif"),
        dpi=100,
        writer=PillowWriter(fps=5),
    )
    print("Done saving images!")
