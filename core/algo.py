import numpy
import torch
from procrustes.permutation import _compute_permutation_hungarian

from helper import timer_func


class Permuter:
    """ "
    Parent class for all the methods
    """

    perm = dict()

    def __init__(self, arch: list[int], model_width) -> None:
        self.arch = arch
        self.model_width = model_width


class ActivationMethod(Permuter):
    cost_matrix = dict()

    def __init__(self, arch, model_width) -> None:
        super().__init__(arch, model_width)

    @timer_func("Activation method")
    def get_permutation(self) -> dict[str, numpy.ndarray]:
        """
        Get's layer wise permutation matrix

        :return: Dictionary of permutation
        :rtype: dict[str, numpy.ndarray]
        """
        # TODO: Compute error of the procrustes method
        if len(self.cost_matrix) == 0:
            raise ValueError("Compute cost matrix first")

        for key in self.cost_matrix.keys():
            self.perm[key] = _compute_permutation_hungarian(self.cost_matrix[key])

        return self.perm

    def evaluate_cost_batch_wise(
        self, model_a: dict[str, torch.Tensor], model_b: dict[str, torch.Tensor]
    ) -> None:
        for key in model_a.keys():
            # TODO: #4 Is this needed? @Adhithyan8
            # if (model_a[key].shape[0] < model_a[key].shape[1]) or (
            #     model_b[key].shape[0] < model_b[key].shape[1]
            # ):
            #     raise Exception("Oh no! Mr dumb ass fucked it up!")

            self.cost_matrix[key] = (
                self.cost_matrix.get(key, 0)
                + (model_a[key].T @ model_b[key]).detach().numpy()
            )


class GreedyAlgorithm(Permuter):
    def __init__(self, arch: list[int], model_width) -> None:
        """
        _summary_

        :param arch: _description_
        :type arch: list[int]
        :param model_width: _description_
        :type model_width: _type_
        """
        super().__init__(arch, model_width)

    def get_permutation(self, model_a, model_b):
        """
        Generate permutation for the weights of model B

        :param model_a: Model A
        :type model_a: torch.nn

        :param model_b: Model B
        :type model_b: torch.nn

        :return: List of permutation of each layer
        :rtype:
        """

        return numpy.zeros((2, 2))
