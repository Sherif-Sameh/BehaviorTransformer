from pathlib import Path

import toml
import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from btransformer.clusterers import KMeansClusterer
from btransformer.loggers import (
    ComposeLogger,
    ConsoleLogger,
    CSVLogger,
)
from btransformer.metrics import (
    AccumulatorMetric,
    ComposeMetric,
    AbsErrorMetric,
)
from btransformer.models import BehaviorTransformerMixedObs, PolicyGPT
from btransformer.trainers import BTransformerTrainer
from btransformer.utils import seed_everything

from dataset import PushTDataset


def main():
    # Load configurations
    seed_everything(seed=0)
    path = Path(__file__).parent / "config/config.toml"
    config = toml.load(path)

    # Initialize dataset and dataloader
    path = Path(__file__).parents[2] / "data/pusht"
    dataset = PushTDataset(path, **config["dataset"])
    dataloader = DataLoader(dataset, **config["dataloader"])

    # Initialize metrics and loggers
    metrics = ComposeMetric([
        AccumulatorMetric("train_loss", red="mean", name="Train Loss"),
        AccumulatorMetric("train_focal_loss", red="mean", name="Train Focal Loss"),
        AccumulatorMetric("train_multi_task_loss", red="mean", name="Train Multi-task Loss"),
        AbsErrorMetric(
            "train_pred_acts",
            "train_targ_acts",
            red="mean",
            name="Abs. Error (Pred. vs Target Actions)",
        ),
    ])
    loggers = ComposeLogger([
        ConsoleLogger(interval=1, filter=None),
        CSVLogger(Path(__file__).parent / "logs/logs.csv", interval=1, filter=None),
    ])

    # Initialize KMeans clusterer and PolicyGPT model
    clusterer = KMeansClusterer(**config["clusterer"])
    clusterer.load(
        Path(__file__).parents[2] / "models/pusht/kmeans.pt"
    )
    policy = PolicyGPT(**config["policy"])

    # Initialize Behavior Transformer model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    img_encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
    model = BehaviorTransformerMixedObs(policy, clusterer, img_encoder).to(device)

    # Initialize trainer and begin training
    ckpt_path = Path(__file__).parents[2] / "models/pusht/btransformer.pt"
    trainer = BTransformerTrainer(model, ckpt_path=ckpt_path, **config["trainer"]["init"])
    trainer.train(
        loss_key=metrics.metrics[0].name,
        metrics=metrics,
        loggers=loggers,
        dataloader=dataloader,
        test_dataloader=None,
        **config["trainer"]["training"],
    )


if __name__ == "__main__":
    main()
