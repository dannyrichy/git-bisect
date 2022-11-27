import copy
from typing import Callable, Sequence, Tuple

import functorch
import numpy
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from config import BIAS, CLASSIFIER, CUDA_AVAILABLE, DEVICE, FEATURES, WEIGHT
from helper import timer_func
from permuter.common import PermDict


def compute_permutation(cost_matrix: numpy.ndarray) -> torch.Tensor:
    """
    Computes the permutation matrix using Hungarian method

    :param cost_matrix: _description_
    :type cost_matrix: numpy.ndarray
    :return: _description_
    :rtype: torch.Tensor
    """
    # solve linear sum assignment problem to get the row/column indices of optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    # make the permutation matrix by setting the corresponding elements to 1
    perm = numpy.zeros(cost_matrix.shape)
    perm[(row_ind, col_ind)] = 1
    return torch.Tensor(perm).to(DEVICE)


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

    @timer_func("Method 1: Computing permutation")
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
            self.perm[key] = compute_permutation(self.cost_matrix[key])

        return self.perm()

    @timer_func("Method 1: Computing cost")
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
                tmp = torch.einsum("ij..., ik... -> jk", model1[key], model2[key])
                tmp = (
                    tmp.detach().cpu().numpy()
                    if CUDA_AVAILABLE
                    else tmp.detach().numpy()
                )
                self.cost_matrix[key] = self.cost_matrix.get(key, 0) + tmp


class WeightMatching(_Permuter):
    def __init__(self, arch: Sequence[str], perm_lookup: dict[str, tuple]) -> None:
        """
        _summary_s

        :param arch: Architecture
        :type arch: list[int]
        """
        super().__init__(arch)
        self.lookup = perm_lookup

    def _initialise_perm(self, m_weights: dict[str, torch.Tensor]) -> None:
        """
        Initialise permutation matrices

        :param m_weights: Model weight dictionary to construct the permutation
        :type m_weights: dict[str, torch.Tensor]
        """
        for key in self.perm.keys:
            self.perm[key] = torch.eye(m_weights[key + "." + WEIGHT].shape[0]).to(
                DEVICE
            )

    def _evaluate_conv_cost(self, layer_name: str, m1_weights, m2_weights):
        _prev_perm, _conv, _next_conv, _next_perm = self.lookup[layer_name]
        _cost_matrix = torch.zeros_like(self.perm[layer_name])

        _cost_matrix += torch.einsum(
            "ij..., jk, lk... -> il",
            m1_weights[_conv + "." + WEIGHT],
            self.perm[_prev_perm]
            if _prev_perm in self.perm.keys
            else torch.eye(m1_weights[_conv + "." + WEIGHT].shape[1]),
            m2_weights[_conv + "." + WEIGHT],
        )
        if _next_conv.startswith(CLASSIFIER):
            _shape = int(
                m2_weights[_next_conv + "." + WEIGHT].shape[1]
                / self.perm[layer_name].size(dim=0)
            )
            _cost_matrix += torch.einsum(
                "i...j, jk, k...l -> il",
                torch.stack(m1_weights[_next_conv + "." + WEIGHT].T.split(_shape)),
                self.perm[_next_perm]
                if _next_perm in self.perm.keys
                else torch.eye(m1_weights[_next_conv + "." + WEIGHT].shape[0]),
                torch.stack(m2_weights[_next_conv + "." + WEIGHT].T.split(_shape)),
            )
        else:
            _cost_matrix += torch.einsum(
                "ji..., jk, kl... -> il",
                m1_weights[_next_conv + "." + WEIGHT],
                self.perm[_next_perm]
                if _next_perm in self.perm.keys
                else torch.eye(m1_weights[_next_conv + "." + WEIGHT].shape[0]),
                m2_weights[_next_conv + "." + WEIGHT],
            )
        _cost_matrix += torch.einsum(
            "i,j -> ij",
            m1_weights[layer_name + "." + BIAS],
            m2_weights[layer_name + "." + BIAS],
        )
        _cost_matrix += torch.einsum(
            "i,j -> ij",
            m1_weights[layer_name + "." + WEIGHT],
            m2_weights[layer_name + "." + WEIGHT],
        )
        _cost_matrix += torch.einsum(
            "i,j -> ij", m1_weights[_conv + "." + BIAS], m2_weights[_conv + "." + BIAS]
        )
        return _cost_matrix

    def _evaluate_linear_cost(
        self, layer_name: str, m1_weights, m2_weights
    ) -> torch.Tensor:
        """
        Cost matrix for linear layer

        :param layer_name: _description_
        :type layer_name: str
        :param m1_weights: _description_
        :type m1_weights: _type_
        :param m2_weights: _description_
        :type m2_weights: _type_
        :return: _description_
        :rtype: torch.Tensor
        """
        _prev_layer, _next_layer = self.lookup[layer_name]
        _cost_matrix = torch.zeros_like(self.perm[layer_name])

        if _prev_layer.startswith(FEATURES):
            # If previous layer is of type features
            _shape = int(
                m2_weights[layer_name + "." + WEIGHT].shape[1]
                / self.perm[_prev_layer].size(dim=0)
            )
            _cost_matrix += torch.einsum(
                "ij..., jk, lk... -> il",
                m1_weights[layer_name + "." + WEIGHT],
                torch.kron(
                    self.perm[_prev_layer].contiguous(), torch.eye(_shape).to(DEVICE)
                ),
                m2_weights[layer_name + "." + WEIGHT],
            ) + torch.einsum(
                "ji..., jk, kl... -> il",
                m1_weights[_next_layer + "." + WEIGHT],
                self.perm[_next_layer],
                m2_weights[_next_layer + "." + WEIGHT],
            )
        else:
            _cost_matrix += torch.einsum(
                "ij..., jk, lk... -> il",
                m1_weights[layer_name + "." + WEIGHT],
                self.perm[_prev_layer]
                if _prev_layer in self.perm.keys
                else torch.eye(m1_weights[layer_name + "." + WEIGHT].shape[1]),
                m2_weights[layer_name + "." + WEIGHT],
            )
            _cost_matrix += torch.einsum(
                "ji..., jk, kl... -> il",
                m1_weights[_next_layer + "." + WEIGHT],
                self.perm[_next_layer]
                if _next_layer in self.perm.keys
                else torch.eye(m1_weights[_next_layer + "." + WEIGHT].shape[0]),
                m2_weights[_next_layer + "." + WEIGHT],
            )

        _cost_matrix += torch.einsum(
            "i,j -> ij",
            m1_weights[layer_name + "." + BIAS],
            m2_weights[layer_name + "." + BIAS],
        )
        return _cost_matrix

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
                if layer_name.startswith(FEATURES):
                    _cost_matrix = self._evaluate_conv_cost(
                        layer_name=layer_name,
                        m1_weights=m1_weights,
                        m2_weights=m2_weights,
                    )

                else:
                    _cost_matrix = self._evaluate_linear_cost(
                        layer_name=layer_name,
                        m1_weights=m1_weights,
                        m2_weights=m2_weights,
                    )

                self.perm[layer_name] = compute_permutation(
                    _cost_matrix.detach().cpu().numpy()
                    if CUDA_AVAILABLE
                    else _cost_matrix.detach().numpy()
                )

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
    """
    Straight Through Estimator

    :param _Permuter: Base Class
    :type _Permuter: _Permuter
    """

    def __init__(self, arch: Sequence[str], perm_lookup) -> None:
        """
        Straight Through Estimator

        :param arch: Architecture
        :type arch: Sequence[str]
        """
        super().__init__(arch)
        self.weight_matching = WeightMatching(arch=arch, perm_lookup=perm_lookup)

    def evaluate_permutation(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
        data_loader: DataLoader,
        permute_model: Callable[
            [torch.nn.Module, dict[str, torch.Tensor]], torch.nn.Module
        ],
    ) -> Tuple[dict[str, torch.Tensor], list]:
        """
        Get permutation matrix for each layer

        :param model1: _description_
        :type model1: torch.nn.Module
        :param model2: _description_
        :type model2: torch.nn.Module
        :param data_loader: _description_
        :type data_loader: DataLoader
        :param permute_model: _description_
        :type permute_model: Callable[ [torch.nn.Module, dict[str, torch.Tensor]], torch.nn.Module ]
        :return: _description_
        :rtype: Tuple[dict[str, torch.Tensor], list]
        """

        # Initialise model_hat
        model_hat = copy.deepcopy(model1)
        loss_arr = list()

        for _ in range(5):
            print(f"Epoch {_} running!")
            for inp, out in data_loader:
                # Finding the permutation
                self.perm = PermDict.from_dict(
                    self.weight_matching.evaluate_permutation(
                        m1_weights=model_hat.state_dict(),
                        m2_weights=model2.state_dict(),
                    )
                )

                # Finding the projected model
                projected_model = permute_model(model2, self.perm())

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
