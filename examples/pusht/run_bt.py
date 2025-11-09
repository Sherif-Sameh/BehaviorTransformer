from argparse import ArgumentParser
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium
import imageio
import toml
import torch
import torchvision.transforms.v2 as tf
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch
from torch import Tensor

from btransformer.clusterers import KMeansClusterer
from btransformer.models import BehaviorTransformerMixedObs, PolicyGPT
from btransformer.models.encoders import ResNetDP
from btransformer.utils import (
    convert_transforms,
    seed_everything,
    unscale_actions,
)

from dataset import PushTDataset


def main(seed: int, n_eps: int):
    seed_everything(seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Recover model configuration
    path = Path(__file__).parent / "config/config.toml"
    config = toml.load(path)
    config["dataset"]["transform"] = convert_transforms(config["dataset"]["transform"])

    # Initialize KMeans clusterer, PolicyGPT and ResNet models
    clusterer = KMeansClusterer(**config["clusterer"])
    clusterer.load(
        Path(__file__).parents[2] / "models/pusht/kmeans.pt"
    )
    policy = PolicyGPT(**config["policy"])
    img_encoder = ResNetDP(**config["encoder"])

    # Initialize and load Behavior Transformer model
    model = BehaviorTransformerMixedObs(policy, clusterer, img_encoder)
    model.load(
        Path(__file__).parents[2] / "models/pusht/btransformer.pt"
    )
    model = model.to(device)

    # Setup observation transformations
    path = Path(__file__).parents[2] / "data/pusht"
    dataset = PushTDataset(path, **config["dataset"])
    dataset.img_tf.transforms[0] = tf.CenterCrop((84, 84))
    acts_low, acts_high = dataset.ACTION_LOW, dataset.ACTION_HIGH
    def img_tf(x: Tensor) -> Tensor:
        return dataset.img_tf(x.permute(2, 0, 1).float() / 255.)
    def prop_tf(x: Tensor) -> Tensor:
        return dataset.prop_tf(x.unsqueeze(0).float()).squeeze(0)
    
    # Create video writer with imageio
    path = Path(__file__).parent / "videos"
    path.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path / "pusht.mp4", fps=60)

    # Create PushT environment with image observations
    env = gymnasium.make(
        "gym_pusht/PushT-v0", obs_type="pixels_agent_pos",  render_mode="rgb_array"
    )
    env = NumpyToTorch(env, device=device)

    # Run policy in environment
    seq_len = config["policy"]["seq_len"]
    for i in range(n_eps):
        # Prepare initial observation sequences
        obs, _ = env.reset(seed=seed)
        img_obs = img_tf(obs["pixels"]).repeat((seq_len, 1, 1, 1)).unsqueeze(0)
        prop_obs = prop_tf(obs["agent_pos"]).repeat((seq_len, 1)).unsqueeze(0)    

        total_rewards = 0.0
        while True:
            # Get policy actions
            action = model.inference(img_obs, prop_obs, deterministic=False)
            action = unscale_actions(action[0], acts_low, acts_high)[-1]

            # Step environment and update observations
            obs, rew, terminated, trucated, _ = env.step(action)
            total_rewards += rew
            img_obs[0, :-1] = img_obs[0, 1:].clone()
            prop_obs[0, :-1] = prop_obs[0, 1:].clone()
            img_obs[0, -1] = img_tf(obs["pixels"])
            prop_obs[0, -1] = prop_tf(obs["agent_pos"])
            writer.append_data(env.render().cpu().numpy())
            
            if terminated or trucated:
                break
        print(f"Episode {i + 1}: Total Rewards = {total_rewards:.2f}\n")
    writer.close()
    print(f"Video saved to {str(path / 'pusht.mp4')}")


if __name__ == "__main__":
    parser = ArgumentParser(prog="Run BeT")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-n", "--n_eps", type=int, default=4)
    
    args = parser.parse_args()
    main(seed=args.seed, n_eps=args.n_eps)