import copy
from typing import Generator

import numpy
import torch
from procrustes.permutation import _compute_permutation_hungarian

from helper import timer_func


class _Permuter:
    """
    Parent class
    """

    perm = dict()

    def __init__(self, arch: list[int]) -> None:
        """
        To store the state of the architecture and common methods

        :param arch: Architecture
        :type arch: list[int]
        :param model_width: # of layers
        :type model_width: int
        """
        self.arch = arch
        self.model_width = len(arch)


class ActMatching(_Permuter):

    cost_matrix = dict()

    def __init__(self, arch: list[int]) -> None:
        """
        Activation method

        :param arch: Architecture
        :type arch: list[int]
        """
        super().__init__(arch)

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


class WeightMatching(_Permuter):
    layer_look_up = dict()

    def __init__(self, arch: list[int]) -> None:
        """
        _summary_

        :param arch: _description_
        :type arch: list[int]
        """
        super().__init__(arch)

    def _initialise_perm(self, m_weights: dict[str, torch.Tensor]) -> None:
        for key, val in m_weights.items():
            layer_name, weight_type = key.split(".")
            if weight_type == "weights":
                self.perm[layer_name] = torch.eye(val.shape[1], dtype=torch.float64)
            _layer_name, _layer_num = layer_name.split("_")
            self.layer_look_up[key] = (
                "_".join([_layer_name, str(int(_layer_num) - 1)]) + "." + weight_type,
                "_".join([_layer_name, str(int(_layer_num) + 1)]) + "." + weight_type,
                (_layer_name, int(_layer_num), weight_type),
            )

    def evaluate_permutation(
        self,
        model1_weights: dict[str, torch.Tensor],
        model2_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        _summary_

        :param model1_weights: _description_
        :type model1_weights: dict[str, torch.Tensor]
        :param model2_weights: _description_
        :type model2_weights: dict[str, torch.Tensor]
        :return: _description_
        :rtype: dict[str, torch.Tensor]
        """
        # TODO: Check convergence criteria, check for loop
        cntr = 0
        self._initialise_perm(model1_weights)
        while cntr < 1000:

            for key in model1_weights.keys():
                _layer_name, _layer_num, weight_type = self.layer_look_up[key]
                if weight_type == "weights":
                    if _layer_num == 1:
                        # Ignoring the permutation in the first layer
                        _cost_matrix = (
                            model1_weights[key] @ model2_weights[key].T
                            + model1_weights[self.layer_look_up[key][1]].T
                            @ self.perm[self.layer_look_up[key][1]]
                            @ model2_weights[self.layer_look_up[key][1]]
                        )
                    elif _layer_num == self.model_width:
                        # Ignoring the permutation in the last layer
                        _cost_matrix = (
                            model1_weights[key]
                            @ self.perm[self.layer_look_up[key][0]]
                            @ model2_weights[key].T
                            + model1_weights[self.layer_look_up[key][1]].T
                            @ model2_weights[self.layer_look_up[key][1]]
                        )
                    else:
                        #  Every other way
                        _cost_matrix = (
                            model1_weights[key]
                            @ self.perm[self.layer_look_up[key][0]]
                            @ model2_weights[key].T
                            + model1_weights[self.layer_look_up[key][1]].T
                            @ self.perm[self.layer_look_up[key][1]]
                            @ model2_weights[self.layer_look_up[key][1]]
                        )
                    self.perm["_".join([_layer_name, str(_layer_num)])] = torch.Tensor(
                        _compute_permutation_hungarian(torch.Tensor.numpy(_cost_matrix))
                    )
            cntr += 1

        return self.perm

    def get_permutation(self) -> dict[str, torch.Tensor]:
        """
        Return the permutation dictionary

        :return: Dictionary of permutation matrices
        :rtype: dict[str, torch.Tensor]
        """
        return self.perm


class STEstimator(_Permuter):
    def __init__(self, arch: list[int]) -> None:
        super().__init__(arch)
        self.weight_matching = WeightMatching(arch=arch)

    def _iterate(
        self, model1: dict[str, torch.Tensor], model2: dict[str, torch.Tensor]
    ):
        model_hat = copy.deepcopy(model1)
        self.weight_matching.evaluate_permutation(model1_weights=model_hat, self.)
