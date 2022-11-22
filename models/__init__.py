"""
Module to hold models
"""

__all__ = ["MLP", "mlp_train", "vgg_train", "mlp_register_hook", "vgg_register_hook", "cifar10_loader"]


from models.mlp import MLP, train as mlp_train, register_hook as mlp_register_hook
from models.vgg import register_hook as vgg_register_hook, train as vgg_train
from models.utils import cifar10_loader
