import copy
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (ACT_PERM, BIAS, DEVICE, LAMBDA_ARRAY, MLP_MODEL1_PATH,
                    MLP_MODEL2_PATH, WEIGHT, WEIGHT_PERM)
from helper import plt_dict, read_file, write_file
from models import MLP, cifar10_loader
from models.mlp import register_hook
from permuter._algo import ActMatching, STEstimator, WeightMatching


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
    permuted_model = MLP().to(DEVICE)

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
) -> np.ndarray:
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
    :rtype: np.ndarray
    """
    loss = [0.0 for _ in range(len(combined_models))]
    for inp, out in data_loader:
        for ix, model in enumerate(combined_models):
            loss[ix] += torch.nn.functional.cross_entropy(
                model(inp.to(DEVICE)), out.to(DEVICE), reduction="sum"
            ).item()

    return np.array(loss) / len(data_loader.dataset)  # type: ignore



def activation_matching() -> dict[str, torch.Tensor]:
    """
    Activation matching code for MLP

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    train_loader, test_loader, _ = cifar10_loader(batch_size=8)
    # TODO: Create checker methods using arch and model_width params
    permuter = ActMatching(arch=[512, 512, 512, 10])

    # Loading individually trained models
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)
    mlp_model2.eval()

    model1_dict, model2_dict = dict(), dict()
    register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
    register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

    # TODO: Time the below two methods and get error value
    # Method 1: Evaluating cost matrix batch wise, values are
    # added element wise
    for inp, lbl in train_loader:
        _ = mlp_model1(inp.to(DEVICE))
        _ = mlp_model2(inp.to(DEVICE))

        # The dictionaries gets erased and updated every time
        permuter.evaluate_permutation(model1_dict, model2_dict)

    # Fetching the permutation
    permutation_dict = permuter.get_permutation()

    return permutation_dict


def weight_matching() -> dict[str, torch.Tensor]:
    """
    weight matching

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)
    mlp_model2.eval()

    weight_matcher = WeightMatching(arch=[512, 512, 512, 10])
    _permutation_dict = weight_matcher.evaluate_permutation(
        model1_weights=mlp_model1.state_dict(), model2_weights=mlp_model2.state_dict()
    )
    return _permutation_dict


def generate_plots(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    act_perm: Optional[dict[str, torch.Tensor]] = None,
    weight_perm: Optional[dict[str, torch.Tensor]] = None,
    ste_perm: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Generate data for plots

    :param model1: Model 1
    :type model1: torch.nn.Module
    :param model2: Model 2
    :type model2: torch.nn.Module
    :param act_perm: Permutation dictionary for Activation Matching, defaults to None
    :type act_perm: Optional[dict[str, torch.Tensor]], optional
    :param weight_perm: Permutation dictionary for Weight Matching, defaults to None
    :type weight_perm: Optional[dict[str, torch.Tensor]], optional
    :param ste_perm: Permutation dictionary for Straight Through Estimator, defaults to None
    :type ste_perm: Optional[dict[str, torch.Tensor]], optional
    :return: Results to use plot
    :rtype: dict[str, dict[str, np.ndarray]]
    """
    # Creating loss_barrier loss function using the above permutation
    # matrix

    train_loader, test_loader, _ = cifar10_loader(batch_size=128)
    result: dict[str, dict[str, np.ndarray]] = dict()

    def _generate_models(_model2: torch.nn.Module) -> dict[str, np.ndarray]:
        """
        Internal function to ensure temporary tensors gets erased

        :param _model2: Model 2
        :type _model2: torch.nn.Module
        :return: Result dictionary
        :rtype: dict[str, np.ndarray]
        """
        _models = list()
        for lam in LAMBDA_ARRAY:
            tmp = combine_models(model1=model1, model2=_model2, lam=lam)
            tmp.eval()
            _models.append(tmp)
        _res = {
            "Train": get_losses(
                data_loader=train_loader,
                combined_models=_models,
            ),
            "Test": get_losses(
                data_loader=test_loader,
                combined_models=_models,
            ),
        }
        return _res

    result["NaiveMatching"] = _generate_models(_model2=model2)
    if act_perm:
        _perm_model = permute_model(model=model2, perm_dict=act_perm)
        _perm_model.eval()
        result["ActivationMatching"] = _generate_models(_model2=_perm_model)
    if weight_perm:
        # TODO: #10 @the-nihilist-ninja Issue with weight matching algo
        _perm_model = permute_model(model=model2, perm_dict=weight_perm)
        _perm_model.eval()
        result["WeightMatching"] = _generate_models(_model2=_perm_model)
    if ste_perm:
        _perm_model = permute_model(model=model2, perm_dict=ste_perm)
        _perm_model.eval()
        result["STEstimator"] = _generate_models(_model2=_perm_model)

    print("Done!")
    return result


def run():

    # if not WEIGHT_PERM.is_file():
    #     weight_perm = weight_matching()
    #     write_file(WEIGHT_PERM, weight_perm)
    # else:
    #     weight_perm = read_file(WEIGHT_PERM)

    # if not ACT_PERM.is_file():
    #     act_perm = activation_matching()
    #     write_file(ACT_PERM, act_perm)
    # else:
    #     act_perm = read_file(ACT_PERM)

    # mlp_model1, mlp_model2 = MLP(), MLP()
    # mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    # mlp_model1.to(DEVICE)
    # mlp_model1.eval()

    # mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    # mlp_model2.to(DEVICE)
    # mlp_model2.eval()

    # results_dict = generate_plots(
    #     model1=mlp_model1, model2=mlp_model2, act_perm=act_perm, weight_perm=weight_perm
    # )

    train_loader, test_loader, _ = cifar10_loader(batch_size=8)

    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)

    ste = STEstimator(arch=[512, 512, 512, 10])
    perm, losses = ste.evaluate_permutation(
        model1=mlp_model1, model2=mlp_model2, data_loader=train_loader, permute_model= permute_model
    )

    results_dict = generate_plots(model1=mlp_model1, model2=mlp_model2, ste_perm=perm)

    # Creating a plot
    plt_dict(results_dict)
