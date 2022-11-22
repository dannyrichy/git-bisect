from pathlib import Path

import numpy
import torch

TIME_FLAG = False

_STASH_PATH = Path("stash")
_STASH_PATH.mkdir(exist_ok=True, parents=True)

MLP_MODEL1_PATH = _STASH_PATH.joinpath("mlp_20221105-105343.pth")
MLP_MODEL2_PATH = _STASH_PATH.joinpath("mlp_20221105-104646.pth")
VGG_MODEL1_PATH = _STASH_PATH.joinpath("vgg_20221122-113333.pth")
VGG_MODEL2_PATH = _STASH_PATH.joinpath("vgg_20221122-114529.pth")

MLP_PERM_PATH = _STASH_PATH.joinpath("mlp_perm")
MLP_PERM_PATH.mkdir(exist_ok=True, parents=True)

WEIGHT_PERM = MLP_PERM_PATH.joinpath("weight_perm.pkl")
ACT_PERM = MLP_PERM_PATH.joinpath("act_perm.pkl")

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

LAMBDA_ARRAY = numpy.linspace(0, 1, 11)
