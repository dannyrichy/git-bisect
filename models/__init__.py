"""
Module to hold models
"""

__all__ = ["MLP", "mlp_train", "register_hook", "cifar10_loader"]

from typing import TypeVar

from models.mlp_model import MLP, mlp_train, register_hook
from models.utils import cifar10_loader
