import numpy
import torch
from procrustes import permutation
from procrustes.utils import compute_error

from core.utils import compute_permutation_hungarian
from helper import timer_func


class Permuter:
    """ "
    Parent class for all the methods
    """

    perm = dict()

    def __init__(self, archi: list[int], model_width):
        self.archi = archi
        self.model_width = model_width


class ActivationMethod(Permuter):
    cost_matrix = dict()

    def __init__(self, archi, model_width) -> None:
        super().__init__(archi, model_width)

    @timer_func("Activation method")
    def get_permuation(self):
        """
        Get's layer wise permutation matrix

        :return: List of permutation matrix
        :rtype: list[numpy.ndarray]
        """
        if len(self.cost_matrix) == 0:
            raise ValueError("Compute cost matrix first")

        for key in self.cost_matrix.keys():
            self.perm[key] = compute_permutation_hungarian(self.cost_matrix[key])

        return self.perm

    def evaluate_cost_batch_wise(
        self, model_a: dict[str, torch.Tensor], model_b: dict[str, torch.Tensor]
    ) -> None:
        for key in model_a.keys():
            # TODO: #4 Is this needed? @Adhithyan8    
            # if (model_a[key].shape[0] < model_a[key].shape[1]) or (
            #     model_b[key].shape[0] < model_b[key].shape[1]
            # ):
            #     raise Exception("Oh no! Mr dumbass fucked it up!")

            self.cost_matrix[key] = (
                self.cost_matrix.get(key, 0) + model_a[key].T @ model_b[key]
            )


class GreedyAlgorithm(Permuter):
    def __init__(self, archi: list[int], model_width) -> None:
        """
        _summary_

        :param archi: _description_
        :type archi: list[int]
        :param model_width: _description_
        :type model_width: _type_
        """
        super().__init__(archi, model_width)

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
