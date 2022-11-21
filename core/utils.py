import copy

import numpy
import torch
from torch.utils.data import DataLoader

from config import DEVICE, LAMBDA_ARRAY

WEIGHT = "weight"
BIAS = "bias"


def permute_model(
    model: torch.nn.Module,
    perm_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    """
    Permute the model with the dictionary

    :param model: Model to be permuted
    :type model: torch.nn.Module
    :param perm_dict: Permutation dictionary
    :type perm_dict: dict[str, torch.Tensor]
    :return: Permuted model
    :rtype: torch.nn.Module
    """
    # Creating model instance to hold the permuted model
    permuted_model = type(model)().to(DEVICE)
    # permuted_model.eval()

    perm_state_dict = permuted_model.state_dict()
    model2_state_dict = model.state_dict()

    for key in perm_state_dict.keys():
        layer_name, weight_type = key.split(".")

        if weight_type == WEIGHT and not layer_name.endswith("1"):
            _layer_name, _layer_num = layer_name.split("_")
            prev_layer_name = "_".join([_layer_name, str(int(_layer_num) - 1)])

            # Considers both column and row permutation if applicable else only column transformation
            # The latter case happens for last layer
            perm_state_dict[key] = (
                (
                    perm_dict[layer_name]
                    @ model2_state_dict[key]
                    @ perm_dict[prev_layer_name].T
                )
                if layer_name in perm_dict
                else model2_state_dict[key] @ perm_dict[prev_layer_name].T
            )
        elif layer_name in perm_dict:
            perm_state_dict[key] = perm_dict[layer_name] @ model2_state_dict[key]

    permuted_model.load_state_dict(perm_state_dict)
    return permuted_model


def combine_models(
    model1: torch.nn.Module, model2: torch.nn.Module, lam: float
) -> torch.nn.Module:
    """
    Combine models using linear interpolation (1-lam)*model1 + lam*model2

    :param model1: Model 1
    :type model1: torch.nn.Module
    :param model2: Model 2
    :type model2: torch.nn.Module
    :param lam: Lambda value in linear interpolation way
    :type lam: float
    :return: Combined model
    :rtype: torch.nn.Module
    """
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


def get_losses(
    data_loader: DataLoader,
    combined_models: list[torch.nn.Module],
) -> numpy.ndarray:
    """
    Generates data for loss barrier plot

    :param data_loader: Data Loader
    :type data_loader: DataLoader
    :param model1: Model 1
    :type model1: torch.nn.Module
    :param model2: Model 2
    :type model2: torch.nn.Module
    :param combined_models: list of combined models for different values of lambda
    :type combined_models: list[torch.nn.Module]
    :return: Loss barrier for combined models
    :rtype: numpy.ndarray
    """
    loss = [0.0 for _ in range(len(combined_models))]
    for inp, out in data_loader:
        for ix, model in enumerate(combined_models):
            loss[ix] += torch.nn.functional.cross_entropy(
                model(inp.to(DEVICE)), out.to(DEVICE), reduction="sum"
            ).item()

    return numpy.array(loss) / len(data_loader.dataset)  # type: ignore
