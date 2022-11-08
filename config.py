from pathlib import Path

import torch

TIME_FLAG = False

MLP_MODEL1_PATH = Path("stash/mlp_20221105-105343.pth")
MLP_MODEL2_PATH = Path("stash/mlp_20221105-104646.pth")

DEVICE = torch.device("cuda:0")
