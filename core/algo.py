import numpy
import torch
from procrustes.permutation import _compute_permutation_hungarian

from helper import timer_func


class _Permuter:
    """
    Parent class
    """

    perm = dict()

    def __init__(self, arch: list[int], model_width: int) -> None:
        """
        To store the state of the architecture and common methods

        :param arch: Architecture
        :type arch: list[int]
        :param model_width: # of layers
        :type model_width: int
        """
        self.arch = arch
        self.model_width = model_width


class ActivationMethod(_Permuter):

    cost_matrix = dict()

    def __init__(self, arch: list[int], model_width: int) -> None:
        """
        Activation method

        :param arch: Architecture
        :type arch: list[int]
        :param model_width: # of layers
        :type model_width: int
        """
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
        self, model1: dict[str, torch.Tensor], model2: dict[str, torch.Tensor]
    ) -> None:
        """
        Computes cost matrix batch wise

        :param model1: Model 1
        :type model1: dict[str, torch.Tensor]
        :param model2: Model 2
        :type model2: dict[str, torch.Tensor]
        """
        for key in model1.keys():
            # TODO: #4 Is this needed? @Adhithyan8
            # if (model_a[key].shape[0] < model_a[key].shape[1]) or (
            #     model_b[key].shape[0] < model_b[key].shape[1]
            # ):
            #     raise Exception("Oh no! Mr dumb ass fucked it up!")

            self.cost_matrix[key] = (
                self.cost_matrix.get(key, 0)
                + (model1[key].T @ model2[key]).detach().numpy()
            )


class GreedyAlgorithm(_Permuter):
    def __init__(
        self, arch: list[int], model_width: int, layer_name: list[str]
    ) -> None:
        """
        _summary_

        :param arch: _description_
        :type arch: list[int]
        :param model_width: _description_
        :type model_width: int
        """
        super().__init__(arch, model_width)

        # Initialise permutation matrix
        self.layer_name = layer_name
        self.perm = {
            layer_name[w]: torch.eye(w, dtype=torch.float64) for w in self.arch
        }

    def evaluate_permutation(
        self, model1_weights: list[torch.Tensor], model2_weights: list[torch.Tensor]
    ):
        """
        Generate permutation for the weights of model B

        :param model1_weights: Model A
        :type model1_weights: torch.nn

        :param model2_weights: Model B
        :type model2_weights: torch.nn

        """
        # TODO: Check convergence criteria, check for loop
        cntr = 0
        while cntr < 1000:
            for i in range(1, self.model_width + 1):
                _cost_matrix = (
                    model1_weights[i]
                    @ self.perm[self.layer_name[i - 1]]
                    @ model2_weights[i].T
                    + model1_weights[i + 1].T
                    @ self.perm[self.layer_name[i + 1]]
                    @ model2_weights[i + 1]
                )
                self.perm[self.layer_name[i]] = torch.Tensor(
                    _compute_permutation_hungarian(torch.Tensor.numpy(_cost_matrix))
                )
            cntr += 1

    def get_permutation(self) -> dict[str, torch.Tensor]:
        """
        Return the permutation dictionary

        :return: Dictionary of permutation matrices
        :rtype: dict[str, torch.Tensor]
        """
        return self.perm
