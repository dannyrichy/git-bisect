import copy
from typing import Sequence

import numpy
import torch
from torch.utils.data import DataLoader

from config import BIAS, DEVICE, WEIGHT


class PermDict:
    def __init__(self, keys: Sequence[str]) -> None:
        """"""
        self.keys = keys
        self._dict: dict = {key: None for key in keys}

    def _check_key(self, key: str):
        if key not in self.keys:
            raise KeyError(f"Key: {key} not a valid key in Permutation dictionary")

    @classmethod
    def from_dict(cls, perm_dict: dict[str, torch.Tensor]):
        tmp = cls(list(perm_dict.keys()))
        for key in perm_dict.keys():
            tmp[key] = perm_dict[key]
        return tmp

    def __setitem__(self, key: str, item: torch.Tensor):
        self._check_key(key)
        self._dict.update({key: item})

    def __getitem__(self, key: str):
        self._check_key(key)
        return self._dict[key]

    def __call__(self) -> dict:
        return self._dict


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


def perm_linear_layer(
    model_sd: dict[str, torch.Tensor],
    perm_dict: dict[str, torch.Tensor],
    layer_look_up: dict[str, str],
) -> dict[str, torch.Tensor]:
    perm_state_dict = copy.deepcopy(model_sd)

    for key in perm_dict.keys():
        _next_key = layer_look_up[key]
        _iter = [key + "." + j for j in (WEIGHT, BIAS)]

        # forward permutation
        for _key in _iter:
            perm_state_dict[_key] = torch.einsum(
                "ij, j... -> i...",
                perm_dict[key],
                perm_state_dict[_key],
            )

        # reverse permutation
        perm_state_dict[_next_key + "." + WEIGHT] = torch.einsum(
            "jk..., ik -> ji...",
            model_sd[_next_key + "." + WEIGHT],
            perm_dict[key],
        )
    return perm_state_dict
