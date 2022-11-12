from pathlib import Path

import numpy
import torch

TIME_FLAG = False

MLP_MODEL1_PATH = Path("stash/mlp_20221105-105343.pth")
MLP_MODEL2_PATH = Path("stash/mlp_20221105-104646.pth")

MLP_PERM_PATH = Path("stash/mlp_perm")

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

LAMBDA_ARRAY = numpy.linspace(0, 1, 11)
