import copy
from typing import Callable, Sequence, Tuple

import functorch
import numpy
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from config import BIAS, CUDA_AVAILABLE, DEVICE, WEIGHT
from helper import timer_func
from permuter.common import PermDict


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

    def __init__(self, arch: Sequence[str]) -> None:
        """
        To store the state of the architecture and common methods

        :param arch: Architecture
        :type arch: list[int]
        :param model_width: # of layers
        :type model_width: int
        """
        self.arch: Sequence[str] = arch
        self.model_width: int = len(arch)
        self.perm: PermDict = PermDict(keys=arch[:-1])


class ActMatching(_Permuter):
    def __init__(self, arch: Sequence[str]) -> None:
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
        if len(self.cost_matrix) == 0:
            raise ValueError(
                "Compute cost matrix first; run evaluate_permutation method!"
            )

        for key in self.cost_matrix.keys():
            self.perm[key] = torch.Tensor(
                compute_permutation(self.cost_matrix[key])
            ).to(DEVICE)

        return self.perm()

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
            if key != self.arch[-1]:
                # tmp = model1[key].T @ model2[key]
                tmp = torch.einsum("ij..., ik... -> jk", model1[key], model2[key])
                tmp = (
                    tmp.detach().cpu().numpy()
                    if CUDA_AVAILABLE
                    else tmp.detach().numpy()
                )
                self.cost_matrix[key] = self.cost_matrix.get(key, 0) + tmp


class WeightMatching(_Permuter):
    def __init__(self, arch: Sequence[str]) -> None:
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
            if weight_type == WEIGHT and layer_name != self.arch[-1]:
                self.perm[layer_name] = torch.eye(val.shape[0]).to(DEVICE)

    def evaluate_permutation(
        self,
        m1_weights: dict[str, torch.Tensor],
        m2_weights: dict[str, torch.Tensor],
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
        _ix = 0
        self._initialise_perm(m1_weights)
        prev_perm = copy.deepcopy(self.perm)
        abs_diff = numpy.inf

        while _ix < 1000 and abs_diff > 5.0:
            abs_diff = 0.0
            for layer_name in self.perm.keys:
                # Getting previous layer name and next layer name
                ix = self.arch.index(layer_name)

                # Adding weights term
                if ix == 0:
                    # Ignoring the permutation in the first layer
                    _cost_matrix = (
                        m1_weights[layer_name + "." + WEIGHT]
                        @ m2_weights[layer_name + "." + WEIGHT].T
                        + m1_weights[self.arch[ix + 1] + "." + WEIGHT].T
                        @ self.perm[self.arch[ix + 1]]
                        @ m2_weights[self.arch[ix + 1] + "." + WEIGHT]
                    )
                elif ix == self.model_width - 2:
                    # Ignoring the permutation in the last layer
                    _cost_matrix = (
                        m1_weights[layer_name + "." + WEIGHT]
                        @ self.perm[self.arch[ix - 1]]
                        @ m2_weights[layer_name + "." + WEIGHT].T
                        + m1_weights[self.arch[ix + 1] + "." + WEIGHT].T
                        @ m2_weights[self.arch[ix + 1] + "." + WEIGHT]
                    )
                else:
                    #  Every other way
                    _cost_matrix = (
                        m1_weights[layer_name + "." + WEIGHT]
                        @ self.perm[self.arch[ix - 1]]
                        @ m2_weights[layer_name + "." + WEIGHT].T
                        + m1_weights[self.arch[ix + 1] + "." + WEIGHT].T
                        @ self.perm[self.arch[ix + 1]]
                        @ m2_weights[self.arch[ix + 1] + "." + WEIGHT]
                    )

                # Adding bias term part
                _cost_matrix += (
                    m1_weights[layer_name + "." + BIAS].unsqueeze(1)
                    @ m2_weights[layer_name + "." + BIAS].unsqueeze(1).T
                )

                self.perm[layer_name] = torch.Tensor(
                    compute_permutation(
                        _cost_matrix.detach().cpu().numpy()
                        if CUDA_AVAILABLE
                        else _cost_matrix.detach().numpy()
                    )
                ).to(DEVICE)
                abs_diff += torch.sum(
                    torch.abs(self.perm[layer_name] - prev_perm[layer_name])
                ).item()
            _ix += 1
            abs_diff = abs_diff
            prev_perm = copy.deepcopy(self.perm)

        return self.perm()

    def get_permutation(self) -> dict[str, torch.Tensor]:
        """
        Return the permutation dictionary

        :return: Dictionary of permutation matrices
        :rtype: dict[str, torch.Tensor]
        """
        return self.perm()


class STEstimator(_Permuter):
    def __init__(self, arch: Sequence[str]) -> None:
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
                self.perm = PermDict.from_dict(
                    self.weight_matching.evaluate_permutation(
                        m1_weights=model_hat.state_dict(),
                        m2_weights=model2.state_dict(),
                    )
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

        return self.perm(), loss_arr
