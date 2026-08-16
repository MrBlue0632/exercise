import argparse
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from model import TrajModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_CHECKPOINT_PATH = Path(__file__).with_name("flowmatching_512.pth")
OUTPUT_PATH = Path(__file__).with_name("fm_samples.png")


@torch.no_grad()
def sample_fm(model, num_samples, num_steps, return_history=False):
    """Generate trajectories by Euler-integrating the learned vector field."""
    model.eval()

    x = torch.randn(
        num_samples,
        model.chunk_size,
        model.action_dim,
        device=DEVICE,
    )
    dt = 1.0 / num_steps
    history = [x.clone()] if return_history else None

    for step in range(num_steps):
        t = torch.full(
            (num_samples,),
            step * dt,
            device=DEVICE,
        )
        velocity = model(x, t)
        x = x + dt * velocity
        if return_history:
            history.append(x.clone())

    if return_history:
        return x, torch.stack(history)
    return x


@torch.no_grad()
def sample_independent_episodes(model, num_samples, num_steps):
    """Generate each displayed trajectory through a separate batch-size-one run."""
    episodes = []
    selected_index = torch.randint(num_samples, ()).item()
    selected_history = None

    for episode_index in range(num_samples):
        if episode_index == selected_index:
            episode, selected_history = sample_fm(
                model,
                num_samples=1,
                num_steps=num_steps,
                return_history=True,
            )
        else:
            episode = sample_fm(model, num_samples=1, num_steps=num_steps)
        episodes.append(episode[0])

    return torch.stack(episodes), selected_history[:, 0], selected_index


def add_gradient_trajectory(axis, trajectory, values, cmap, norm, alpha=1.0):
    """Draw one 2D trajectory with a color assigned to every line segment."""
    points = trajectory[:, :2].reshape(-1, 1, 2)
    segments = torch.cat([points[:-1], points[1:]], dim=1).numpy()
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        array=values[:-1],
        linewidth=1.5,
        alpha=alpha,
    )
    axis.add_collection(collection)


def plot_samples(samples, episode_history, selected_index, output_path):
    samples = samples.cpu()
    episode_history = episode_history.cpu()
    cmap = plt.colormaps["viridis_r"]  # yellow at the start, purple at the end

    figure, (trajectory_ax, endpoint_ax) = plt.subplots(1, 2, figsize=(12, 5))

    chunk_time = torch.linspace(0, 1, samples.shape[1])
    chunk_norm = Normalize(vmin=0, vmax=1)
    for trajectory in samples:
        add_gradient_trajectory(
            trajectory_ax,
            trajectory,
            chunk_time,
            cmap,
            chunk_norm,
            alpha=0.35,
        )
    trajectory_ax.autoscale_view()
    trajectory_ax.set_title("Generated action chunks")
    trajectory_ax.set_xlabel("action x")
    trajectory_ax.set_ylabel("action y")
    trajectory_ax.set_aspect("equal", adjustable="box")
    trajectory_ax.grid(alpha=0.3)
    figure.colorbar(
        ScalarMappable(norm=chunk_norm, cmap=cmap),
        ax=trajectory_ax,
        label="action chunk time: start -> end",
    )

    generation_time = torch.linspace(0, 1, episode_history.shape[0])
    generation_norm = Normalize(vmin=0, vmax=1)
    for state, generation_t in zip(episode_history, generation_time):
        endpoint_ax.plot(
            state[:, 0],
            state[:, 1],
            color=cmap(generation_norm(generation_t)),
            alpha=0.2,
            linewidth=1.0,
        )
    endpoint_ax.set_title(f"Episode {selected_index} during generation")
    endpoint_ax.set_xlabel("action x")
    endpoint_ax.set_ylabel("action y")
    endpoint_ax.set_aspect("equal", adjustable="box")
    endpoint_ax.grid(alpha=0.3)
    figure.colorbar(
        ScalarMappable(norm=generation_norm, cmap=cmap),
        ax=endpoint_ax,
        label="generation time: noise -> trajectory",
    )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    args = parser.parse_args()

    weights = torch.load(args.checkpoint, map_location=DEVICE, weights_only=True)
    hidden_dim = weights["net.0.weight"].shape[0]
    model = TrajModel(hidden_dim=hidden_dim).to(DEVICE)
    model.load_state_dict(weights)
    print(f"Loaded checkpoint: {args.checkpoint} (hidden_dim={hidden_dim})")

    samples, episode_history, selected_index = sample_independent_episodes(
        model,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
    )
    plot_samples(samples, episode_history, selected_index, OUTPUT_PATH)
    torch.save(samples.cpu(), Path(__file__).with_name("fm_samples.pt"))

    print(f"Saved plot to: {OUTPUT_PATH}")
    print(f"Saved samples to: {Path(__file__).with_name('fm_samples.pt')}")


if __name__ == "__main__":
    main()
