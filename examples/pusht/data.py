"""
Analysis of the raw PushT dataset observation and action statistics.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch import Tensor


def get_obs_stats(path: Path) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute and return the mean and std values for both image and proprioceptive observations.
    
    Args:
        path: Path for storing PushT dataset.
    
    Returns:
        tuple
        - img_mean: (3,) Tensor of mean instensity values per channel.
        - prop_mean: (2,) Tensor of mean values for proprioceptive observations.
    """
    # Initialize dataset and tensors for running stats
    dataset = LeRobotDataset("lerobot/pusht", root=path)
    img_mean = torch.zeros(3, dtype=torch.float64)
    img_M2 = torch.zeros(3, dtype=torch.float64)
    prop_mean = torch.zeros(2, dtype=torch.float64)
    prop_M2 = torch.zeros(2, dtype=torch.float64)

    # Compute running stats using Welford's algorithm
    count = 0
    for i in range(dataset.num_frames):
        batch = dataset[i]
        img = batch["observation.image"].to(torch.float64).mean(dim=(1, 2))
        prop = batch["observation.state"].to(torch.float64)
        count += 1

        # Update running stats
        delta_img = img - img_mean
        img_mean += delta_img / count
        img_M2 += delta_img * (img - img_mean)
        delta_prop = prop - prop_mean
        prop_mean += delta_prop / count
        prop_M2 += delta_prop * (prop - prop_mean)

    # Compute final standard deviations
    img_std = torch.sqrt(img_M2 / (count - 1))
    prop_std = torch.sqrt(prop_M2 / (count - 1))
    return img_mean, img_std, prop_mean, prop_std


def plot_actions(path: Path) -> None:
    # Initialize dataset and action tensor
    dataset = LeRobotDataset("lerobot/pusht", root=path)
    actions = torch.zeros((dataset.num_frames, 2), dtype=torch.float64)
    
    # Get all actions
    for i in range(dataset.num_frames):
        actions[i] = dataset[i]["action"]
    
    # Plot actions onto a scatter plot
    plt.scatter(actions[:, 0], actions[:, 1], marker="x", s=10)
    plt.title("PushT Action Distribution")
    plt.xlabel("Action (x)")
    plt.ylabel("Action (y)")
    plt.xlim([0, 512])
    plt.ylim([0, 512])
    plt.grid(True)
    plt.show()


def main():
    path = Path(__file__).parents[2] / "data/pusht"
    
    # Get and print observation stats
    img_mean, img_std, prop_mean, prop_std = get_obs_stats(path)
    print(f"Image observations mean: {img_mean}")
    print(f"Image observations std: {img_std}")
    print(f"Proprioceptive observations mean: {prop_mean}")
    print(f"Proprioceptive observations std: {prop_std}")

    # Plot action distribution
    plot_actions(path)


if __name__ == "__main__":
    main()