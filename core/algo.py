import torch

from procrustes.permutation import permutation


class Algorithm:
    """"
    Implementation for MLP
    """

    def __init__(self, dim, model_width):
        self.loss = None
        self.dim = dim
        self.model_width = model_width


class ActivationMethod(Algorithm):
    def __init__(self, dim, model_width):
        super().__init__(dim, model_width)

    def _layer_wise(self, act_a, act_b):
        """
        Model B is considered to be permuted

        :param act_a: Activation of Model a layer in the model
        :type act_a: torch.Tensor

        :param act_b: Activation of Model b layer in the model
        :type act_b: torch.Tensor

        :return: permutation matrix for the corresponding layer
        :rtype:
        """
