from functools import partial
from typing import Callable, Union

import numpy
import torch

from models.mlp_model import MLP


def _naive_combine_models(model1: MLP, model2: MLP, lam: float = 0.5) -> MLP:
    """
    Combine models without permutation

    :param model1: Model 1
    :type model1: MLP
    :param model2: Model 2
    :type model2: MLP
    :param lam: lambda value to combine models by, defaults to 0.5
    :type lam: float, optional
    :return: Combined models
    :rtype: MLP
    """
    model3 = MLP()
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    model3_state_dict = model3.state_dict()

    for key in model3_state_dict.keys():
        model3_state_dict[key] = (1 - lam) * model1_state_dict[
            key
        ] + lam * model2_state_dict[key]

    model3.load_state_dict(model3_state_dict)

    return model3


def _combine_models(
    model1: MLP,
    model2: MLP,
    perm_dict: dict[str, numpy.ndarray],
    lam: float = 0.5,
) -> MLP:
    """
    Combines models model1 and model2 as (1-lam)*model1 + lam*model2

    :param model1: Model 1
    :type model1: MLP
    :param model2: Model 2
    :type model2: MLP
    :param perm_dict: Permutation dictionary
    :type perm_dict: dict[str, numpy.ndarray]
    :param lam: lambda value to combine models by, defaults to 0.5
    :type lam: float, optional
    :return: combined model
    :rtype: MLP
    """
    model3 = MLP()
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    model3_state_dict = model3.state_dict()

    for key in model3_state_dict.keys():
        model3_state_dict[key] = (1 - lam) * model1_state_dict[
            key
        ] + lam * torch.Tensor(perm_dict[key.split(".")[0]]) @ model2_state_dict[key]

    model3.load_state_dict(model3_state_dict)

    return model3


def loss_barrier(
    model1: MLP,
    model2: MLP,
    lambda_list: Union[numpy.ndarray, list[float]],
    perm_dict: dict[str, numpy.ndarray],
) -> Callable[[torch.Tensor, torch.Tensor], dict[str, numpy.ndarray]]:
    """
    Returns function to calculate loss barrier for all values in lambda_list

    :param model1: Model 1
    :type model1: MLP
    :param model2: Model 2 whose permutation is taken
    :type model2: MLP
    :param lambda_list: list of lambda values to combine models
    :type lambda_list: Union[numpy.ndarray, list[float]]
    :param perm_dict: Permutation dictionary
    :type perm_dict: dict[str, numpy.ndarray]
    :return: Function analogous to loss
    :rtype: Callable[[torch.Tensor, torch.Tensor], dict[str,numpy.ndarray]]
    """
    # TODO: Check if the following loss function is correct
    # Should it get the logits or the softmax output
    _combined_model = partial(_combine_models, model1, model2, perm_dict)
    _naive_combined_model = partial(_naive_combine_models, model1, model2)

    def get_list_loss_barrier(
        inp: torch.Tensor, out: torch.Tensor
    ) -> dict[str, numpy.ndarray]:
        _sum_losses = 0.5 * (
            torch.nn.functional.cross_entropy(model1(inp), out)
            + torch.nn.functional.cross_entropy(model2(inp), out)
        )
        return {
            "Activation matching": numpy.array(
                [
                    (
                        torch.nn.functional.cross_entropy(
                            _combined_model(_lam_val)(inp), out
                        )
                        - _sum_losses
                    ).item()
                    for _lam_val in lambda_list
                ]
            ),
            "Naive matching": numpy.array(
                [
                    (
                        torch.nn.functional.cross_entropy(
                            _naive_combined_model(_lam_val)(inp), out
                        )
                        - _sum_losses
                    ).item()
                    for _lam_val in lambda_list
                ]
            ),
        }

    return get_list_loss_barrier
