import copy
from typing import Callable, Tuple

import functorch
import numpy
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from config import BIAS, CUDA_AVAILABLE, DEVICE, WEIGHT
from helper import timer_func


def compute_permutation(cost_matrix: numpy.ndarray) -> numpy.ndarray:
    # solve linear sum assignment problem to get the row/column indices of optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # make the permutation matrix by setting the corresponding elements to 1
    perm = numpy.zeros(cost_matrix.shape)
    perm[(row_ind, col_ind)] = 1
    return perm


class _Permuter:
    """
    Parent class for permutation method
    """

    def __init__(self, arch: list[int]) -> None:
        """
        To store the state of the architecture and common methods

        :param arch: Architecture
        :type arch: list[int]
        :param model_width: # of layers
        :type model_width: int
        """
        self.arch: list[int] = arch
        self.model_width: int = len(arch)
        self.perm: dict[str, torch.Tensor] = dict()
        self.layer_look_up: dict[str, tuple[str, str, tuple[str, int]]] = dict()


class ActMatching(_Permuter):
    def __init__(self, arch: list[int]) -> None:
        """
        Activation method

        :param arch: Architecture
        :type arch: list[int]
        """
        super().__init__(arch)
        self.cost_matrix: dict[str, numpy.ndarray] = dict()

    @timer_func("Activation method")
    def get_permutation(self) -> dict[str, torch.Tensor]:
        """
        Get's layer wise permutation matrix

        :return: Dictionary of permutation
        :rtype: dict[str, numpy.ndarray]
        """
        # TODO: Compute error of the procrustes method
        if len(self.cost_matrix) == 0:
            raise ValueError("Compute cost matrix first")

        for key in self.cost_matrix.keys():
            self.perm[key] = torch.Tensor(
                compute_permutation(self.cost_matrix[key])
            ).to(DEVICE)

        return self.perm

    def evaluate_permutation(
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
            tmp = model1[key].T @ model2[key]
            tmp = tmp.detach().cpu().numpy() if CUDA_AVAILABLE else tmp.detach().numpy()
            self.cost_matrix[key] = self.cost_matrix.get(key, 0) + tmp


class WeightMatching(_Permuter):
    def __init__(self, arch: list[int]) -> None:
        """
        _summary_

        :param arch: Architecture
        :type arch: list[int]
        """
        super().__init__(arch)

    def _initialise_perm(self, m_weights: dict[str, torch.Tensor]) -> None:
        """
        Initialise permutation matrices

        :param m_weights: Model weight dictionary to construct the permutation
        :type m_weights: dict[str, torch.Tensor]
        """
        for key, val in m_weights.items():
            layer_name, weight_type = key.split(".")
            if weight_type == WEIGHT:
                _layer_name, _layer_num = layer_name.split("_")
                if int(_layer_num) != self.model_width:
                    self.perm[layer_name] = torch.eye(val.shape[0]).to(DEVICE)
                    self.layer_look_up[key] = (
                        "_".join([_layer_name, str(int(_layer_num) - 1)]),
                        "_".join([_layer_name, str(int(_layer_num) + 1)]),
                        (_layer_name, int(_layer_num)),
                    )

    def evaluate_permutation(
        self,
        model1_weights: dict[str, torch.Tensor],
        model2_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Evaluate permutation

        :param model1_weights: Model 1 weights
        :type model1_weights: dict[str, torch.Tensor]
        :param model2_weights: Model 2 weights
        :type model2_weights: dict[str, torch.Tensor]
        :return: Permutation dictionary
        :rtype: dict[str, torch.Tensor]
        """
        # TODO: Check if model is in DEVICE
        cntr = 0
        self._initialise_perm(model1_weights)
        prev_perm = copy.deepcopy(self.perm)
        abs_diff = numpy.inf

        while cntr < 1000 and abs_diff > 5.0:
            abs_diff = 0.0
            for key in model1_weights.keys():
                if key in self.layer_look_up:
                    _layer_name, _layer_num = self.layer_look_up[key][2]
                    if _layer_num == 1:
                        # Ignoring the permutation in the first layer
                        _cost_matrix = (
                            model1_weights[key] @ model2_weights[key].T
                            + model1_weights[
                                self.layer_look_up[key][1] + "." + WEIGHT
                            ].T
                            @ self.perm[self.layer_look_up[key][1]]
                            @ model2_weights[self.layer_look_up[key][1] + "." + WEIGHT]
                        )
                    elif _layer_num == self.model_width - 1:
                        # Ignoring the permutation in the last layer
                        _cost_matrix = (
                            model1_weights[key]
                            @ self.perm[self.layer_look_up[key][0]]
                            @ model2_weights[key].T
                            + model1_weights[
                                self.layer_look_up[key][1] + "." + WEIGHT
                            ].T
                            @ model2_weights[self.layer_look_up[key][1] + "." + WEIGHT]
                        )
                    else:
                        #  Every other way
                        _cost_matrix = (
                            model1_weights[key]
                            @ self.perm[self.layer_look_up[key][0]]
                            @ model2_weights[key].T
                            + model1_weights[
                                self.layer_look_up[key][1] + "." + WEIGHT
                            ].T
                            @ self.perm[self.layer_look_up[key][1]]
                            @ model2_weights[self.layer_look_up[key][1] + "." + WEIGHT]
                        )

                    # Adding bias term part
                    _cost_matrix += (
                        model1_weights[
                            "_".join([_layer_name, str(_layer_num)]) + "." + BIAS
                        ].unsqueeze(1)
                        @ model2_weights[
                            "_".join([_layer_name, str(_layer_num)]) + "." + BIAS
                        ]
                        .unsqueeze(1)
                        .T
                    )

                    self.perm["_".join([_layer_name, str(_layer_num)])] = torch.Tensor(
                        compute_permutation(
                            _cost_matrix.detach().cpu().numpy()
                            if CUDA_AVAILABLE
                            else _cost_matrix.detach().numpy()
                        )
                    ).to(DEVICE)
                    abs_diff += torch.sum(
                        torch.abs(
                            self.perm["_".join([_layer_name, str(_layer_num)])]
                            - prev_perm["_".join([_layer_name, str(_layer_num)])]
                        )
                    ).item()
            cntr += 1
            abs_diff = abs_diff
            prev_perm = copy.deepcopy(self.perm)

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
        """
        Straight Through Estimator

        :param arch: Architecture
        :type arch: list[int]
        """
        super().__init__(arch)
        self.weight_matching = WeightMatching(arch=arch)

    def evaluate_permutation(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        data_loader: DataLoader,
        permute_model: Callable,
    ) -> Tuple[dict[str, torch.Tensor], list]:
        """
        Get permutation matrix for each layer

        :param model1: _description_
        :type model1: torch.nn.Module
        :param model2: _description_
        :type model2: torch.nn.Module
        :param data_loader: _description_
        :type data_loader: DataLoader
        :return: _description_
        :rtype: dict[str, torch.Tensor]
        """

        # TODO: Check {->} caution: Ensure models are not in eval mode

        # Initialise model_hat
        model_hat = copy.deepcopy(model1)
        loss_arr = list()

        for _ in range(1):
            for inp, out in data_loader:
                # Finding the permutation
                self.perm = self.weight_matching.evaluate_permutation(
                    model1_weights=model_hat.state_dict(),
                    model2_weights=model2.state_dict(),
                )

                # Finding the projected model
                projected_model = permute_model(model=model2, perm_dict=self.perm)

                func, params_1 = functorch.make_functional(model1)
                _, params_hat = functorch.make_functional(model_hat)
                _, params_proj = functorch.make_functional(projected_model)

                params_merged = ()
                for p_1, p_hat, p_proj in zip(params_1, params_hat, params_proj):
                    params_merged += tuple(
                        (
                            0.5
                            * (
                                p_1.detach()
                                + (p_proj.detach() + (p_hat - p_hat.detach()))
                            )
                        ).unsqueeze(0)
                    )

                # Defining the optimizer
                optim = torch.optim.SGD(params=params_hat, lr=0.01, momentum=0.9)
                logits = func(params_merged, inp.to(DEVICE))
                loss = cross_entropy(logits, out.to(DEVICE))
                loss_arr.append(loss.item())
                loss.backward()
                optim.step()
                optim.zero_grad()
                model_hat.load_state_dict(
                    {name: param for name, param in zip(func.param_names, params_hat)}
                )

        return self.perm, loss_arr
