from pathlib import Path

import toml
import torch
from torch import Tensor

from btransformer.clusterers import KMeansClusterer
from btransformer.utils import seed_everything

from dataset import PushTDataset


def get_all_actions(path: Path) -> Tensor:
    """Gather all actions from dataset
    
    Args:
        path: Path to the PushT dataset.
    
    Returns:
        (N, 2) Tensor containing all actions in the dataset.
    """
    dataset = PushTDataset(path, seq_len=1, stride=1)
    actions = torch.zeros((len(dataset), 2), dtype=torch.float32)
    for i in range(len(dataset)):
        _, _, a = dataset[i]
        actions[i] = a[0]
    return actions


def main():
    # Load configurations
    seed_everything(seed=0)
    path = Path(__file__).parent / "config/config.toml"
    config = toml.load(path)
    
    # Get all actions to initialize clusterer with
    path = Path(__file__).parents[2] / "data/pusht"
    actions = get_all_actions(path)
    
    # Initialize KMeans clusterer with dataset actions
    clusterer = KMeansClusterer(**config["clusterer"])
    clusterer(actions)
    
    # Save KMeans clusterer's configuration
    clusterer.save(
        Path(__file__).parents[2] / "models/pusht/kmeans.pt"
    )


if __name__ == "__main__":
    main()
