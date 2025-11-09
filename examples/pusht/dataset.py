"""
Contains PushT dataset wrapper class around the LeRobotDataset.
"""

from pathlib import Path
from functools import partial

import torch
import torchvision.transforms.v2 as tf
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch import Tensor

from btransformer.utils import normalize_prop_obs, rescale_actions


class PushTDataset:
    """Wrapper class around the PushT LeRobot dataset.

    The wrapper provides the following additional functionalities:
    1) Extraction of sequences (with a variable stride) instead of single samples.
    2) Extracts observations and actions from the dataset dict return format.
    3) Handles moving tensors to appropriate device.
    4) Normalization of image and proprioceptive observations and rescaling of actions.  
    
    Args:
        path: Path for storing PushT dataset.
        transform: Image transform to apply to the image observations.
        seq_len: Length of sequences to extract from dataset. Defaults to 1.
        stride: Stride for sampling sequences. Defaults to 1.
        device: Device to move tensors to.
    """
    PROP_MEAN = (229.1110, 293.3112)
    PROP_STD = (101.8567,  96.4914)
    ACTION_LOW = (0.0, 0.0)
    ACTION_HIGH = (512.0, 512.0)

    def __init__(
        self,
        path: Path,
        transform: tf.Transform = tf.Identity(),
        seq_len: int = 1,
        stride: int = 1,
        device: str = "cpu",
    ):
        assert not path.is_file()
        assert seq_len > 0 and stride > 0
        assert device in ["cpu", "cuda"], "Device must be either 'cpu' or 'cuda'."
        device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.seq_len = seq_len
        self.stride = stride
        self.device = torch.device(device)
        self.dataset = LeRobotDataset("lerobot/pusht", root=path)
        
        # Set transforms for observations and actions
        self.img_tf = transform
        self.prop_tf = partial(
            normalize_prop_obs, mean=self.PROP_MEAN, std=self.PROP_STD
        )
        self.action_tf = partial(
            rescale_actions, acts_low=self.ACTION_LOW, acts_high=self.ACTION_HIGH
        )

        # Calculate dataset metadata needed for sampling
        eps_lens = self._get_episode_lens()
        assert eps_lens.sum() == self.dataset.num_frames
        eps_n_seqs = (eps_lens - seq_len) // stride + 1
        self.eps_start_idxs = torch.cumsum(torch.cat([torch.zeros(1), eps_lens[:-1]]), 0).long()
        self.cum_eps_n_seqs = torch.cat([torch.cumsum(eps_n_seqs, 0), torch.zeros(1)]).long()
        self.total_n_seqs = eps_n_seqs.sum().item()
        
    def __len__(self) -> int:
        """Get length of dataset in terms of number of available sequences.

        Note: Only valid sequences whose experiences span a single episode are counted.

        Returns:
            Length of dataset.
        """
        return self.total_n_seqs
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Get a sequence of samples from dataset at specified index.

        Args:
            idx: Index of sequence to retrieve from dataset.

        Returns:
            tuple
            - img_obs: (seq_len, 3, 96, 96) Tensor of image observations.
            - prop_obs: (seq_len, 2) Tensor of proprioceptive observations.
            - actions: (seq_len, 2) Tensor of actions.
        """
        # Find starting index for sequence
        eps_idx = torch.nonzero(self.cum_eps_n_seqs > idx)[0, 0].item()
        eps_seq_idx = idx - self.cum_eps_n_seqs[eps_idx - 1].item()
        start_idx = self.eps_start_idxs[eps_idx].item() + eps_seq_idx * self.stride
        
        # Allocate output tensors
        img_obs = torch.empty((self.seq_len, 3, 96, 96), dtype=torch.float32)
        prop_obs = torch.empty((self.seq_len, 2), dtype=torch.float32)
        actions = torch.empty((self.seq_len, 2), dtype=torch.float32)

        # Extract each sample separately (LeRobotDataset does not support slice indexing)        
        for i in range(self.seq_len):
            data = self.dataset[start_idx + i]
            img_obs[i] = data["observation.image"]
            prop_obs[i] = data["observation.state"]
            actions[i] = data["action"]
        
        # Tranform all tensors with their respective transformations
        img_obs = self.img_tf(img_obs.to(self.device))
        prop_obs = self.prop_tf(prop_obs.to(self.device))
        actions = self.action_tf(actions.to(self.device))
        return img_obs, prop_obs, actions
    
    def _get_episode_lens(self) -> Tensor:
        """Get the lengths of each individual episode in the PushT dataset.
        
        Returns:
            (n_episodes,) Tensor containing the length of each episode in the dataset.
        """
        eps_idx = -1
        eps_lens = []
        for i in range(len(self.dataset)):
            idx = self.dataset[i]["episode_index"].item()
            if idx == eps_idx:
                eps_lens[-1] += 1
            else:
                eps_lens.append(1)
                eps_idx += 1
        eps_lens = torch.tensor(eps_lens).long()
        return eps_lens


def main():
    path = Path(__file__).parents[2] / "data/pusht"
    dataset = PushTDataset(path, seq_len=10, stride=1, device="cuda")
    img_obs, prop_obs, actions = dataset[10]
    print(img_obs.shape, prop_obs.shape, actions.shape)


if __name__ == "__main__":
    main()