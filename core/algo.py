import torch

from core.utils import frobenius_inner_product


class Algorithm:
    def __init__(self, dim):
        self.loss = None
        self.dim = dim
        self.permutation = torch.zeros((self.dim, self.dim))

    # Method 1: Matching activations
    def match_activation(self, activation_1, activation_2):
        """

        """
        maximiser = frobenius_inner_product(self.permutation, torch.matmul(activation_1, torch.transpose(activation_2)))

