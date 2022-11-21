"""
Module to hold models
"""

__all__ = ["MLP", "mlp_train", "register_hook", "cifar10_loader"]


from models.mlp import MLP, mlp_train, register_hook
from models.utils import cifar10_loader
