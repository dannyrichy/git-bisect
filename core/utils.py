import copy
from typing import Union

import numpy
import torch

from config import DEVICE


def permute_model(model: torch.nn.Module, perm_dict: dict[str, torch.Tensor]) -> torch.nn.Module:
    """
    Permute the model with the dictionary

    :param model: Model to be permuted
    :type model: torch.nn.Module
    :param perm_dict: Permutation dictionary
    :type perm_dict: dict[str, torch.Tensor]
    :return: Permuted model
    :rtype: torch.nn.Module
    """
    permuted_model = copy.deepcopy(model).to(DEVICE)
    perm_state_dict = permuted_model.state_dict()
    model2_state_dict = model.state_dict()
    
    for key in perm_state_dict.keys():
        layer_name, weight_type = key.split(".")
        if layer_name in perm_dict.keys():
            if weight_type == "weight" and not layer_name.endswith("1"):
                _layer_name, _layer_num = layer_name.split("_")
                prev_layer_name = "_".join([_layer_name, str(int(_layer_num) - 1)])
                perm_state_dict[key] = (
                    perm_dict[layer_name]
                    @ model2_state_dict[key]
                    @ perm_dict[prev_layer_name].T
                )
            else:
                perm_state_dict[key] = perm_dict[layer_name] @ model2_state_dict[key]
    permuted_model.load_state_dict(perm_state_dict)
    return permuted_model


def combine_models(model1: torch.nn.Module, model2: torch.nn.Module, lam: float) -> torch.nn.Module:
    # Creating dummy model
    model3 = copy.deepcopy(model1).to(DEVICE)
    model3_state_dict = model3.state_dict()
    model1.to(DEVICE)
    model2.to(DEVICE)
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    for key in model3_state_dict.keys():
        model3_state_dict[key] = (1 - lam) * state_dict1[key] + lam * state_dict2[key]

    model3.load_state_dict(model3_state_dict)
    return model3


def loss_barrier(
    data_loader: torch.utils.data.DataLoader,  # type: ignore
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    combined_models: list[torch.nn.Module],
) -> numpy.ndarray:
    """
    Generates data for loss barrier plot

    :param data_loader: _description_
    :type data_loader: torch.utils.data.DataLoader
    :param model1: _description_
    :type model1: torch.nn.Module
    :param model2: _description_
    :type model2: torch.nn.Module
    :param combined_models: _description_
    :type combined_models: list[torch.nn.Module]
    :return: _description_
    :rtype: dict[str, numpy.ndarray]
    """
    loss_barrier_dict = dict()
    counter = 0.0
    loss = [0.0 for _ in range(len(combined_models))]
    for inp, out in data_loader:
        _sum_losses = (
            0.5
            * (
                torch.nn.functional.cross_entropy(
                    model1(inp.to(DEVICE)), out.to(DEVICE)
                )
                + torch.nn.functional.cross_entropy(
                    model2(inp.to(DEVICE)), out.to(DEVICE)
                )
            ).item()
        )

        for ix, model in enumerate(combined_models):
            loss[ix] += (
                torch.nn.functional.cross_entropy(
                    model(inp.to(DEVICE)), out.to(DEVICE)
                ).item()
                - _sum_losses
            )

        counter += 1.0

    return numpy.array(loss) / counter
