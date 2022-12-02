from copy import copy
from typing import Optional

import numpy as np
import torch

from config import (
    BIAS,
    DEVICE,
    LAMBDA_ARRAY,
    MLP_MODEL1_PATH,
    MLP_MODEL2_PATH,
    MLP_PERM_PATH,
    MLP_RESULTS_PATH,
    WEIGHT,
)
from helper import plt_dict, read_file, write_file
from models import MLP, cifar10_loader
from models.mlp import INDEX_LAYER, LAYER_NAMES, WEIGHT_PERM_LOOKUP, register_hook
from permuter._algo import ActMatching, STEstimator, WeightMatching
from permuter.common import combine_models, get_losses, perm_linear_layer

WEIGHT_PERM = MLP_PERM_PATH.joinpath("weight_perm.pkl")
ACT_PERM = MLP_PERM_PATH.joinpath("act_perm.pkl")
STE_PERM = MLP_PERM_PATH.joinpath("ste_perm.pkl")


def permute_model(
    model: torch.nn.Module,
    perm_dict: dict[str, torch.Tensor],
    width:int=512
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
    permuted_model = MLP(WIDTH=width).to(DEVICE)
    perm_state_dict = perm_linear_layer(
        model_sd=model.state_dict(), perm_dict=perm_dict, layer_look_up=INDEX_LAYER
    )
    permuted_model.load_state_dict(perm_state_dict)
    return permuted_model


def activation_matching() -> dict[str, torch.Tensor]:
    """
    Activation matching code for MLP

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    print("Running Activation matching!")
    train_loader, test_loader, _ = cifar10_loader(batch_size=512)
    # TODO: Create checker methods using arch and model_width params
    permuter = ActMatching(arch=LAYER_NAMES)

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
    print("Running Weight Matching!")
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)
    mlp_model2.eval()

    weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
    _permutation_dict = weight_matcher.evaluate_permutation(
        m1_weights=mlp_model1.state_dict(), m2_weights=mlp_model2.state_dict()
    )
    return _permutation_dict


def ste_matching() -> dict[str, torch.Tensor]:
    print("Running STEstimator")
    train_loader, test_loader, _ = cifar10_loader(batch_size=256)
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)

    ste = STEstimator(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
    perm, losses = ste.evaluate_permutation(
        model1=mlp_model1,
        model2=mlp_model2,
        data_loader=train_loader,
        permute_model=permute_model,
    )
    return perm


def generate_plots(
    model1: torch.nn.Module,
    model2: torch.nn.Module,    
    act_perm: Optional[dict[str, torch.Tensor]] = None,
    weight_perm: Optional[dict[str, torch.Tensor]] = None,
    ste_perm: Optional[dict[str, torch.Tensor]] = None,
    width:int=512,
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
        _perm_model = permute_model(model=model2, perm_dict=act_perm, width=width)
        _perm_model.eval()
        result["ActivationMatching"] = _generate_models(_model2=_perm_model)
    if weight_perm:
        _perm_model = permute_model(model=model2, perm_dict=weight_perm, width=width)
        _perm_model.eval()
        result["WeightMatching"] = _generate_models(_model2=_perm_model)
    if ste_perm:
        _perm_model = permute_model(model=model2, perm_dict=ste_perm, width=width)
        _perm_model.eval()
        result["STEstimator"] = _generate_models(_model2=_perm_model)

    print("Done!")
    return result


def run():

    if not WEIGHT_PERM.is_file():
        weight_perm = weight_matching()
        write_file(WEIGHT_PERM, weight_perm)
    else:
        weight_perm = read_file(WEIGHT_PERM)

    if not ACT_PERM.is_file():
        act_perm = activation_matching()
        write_file(ACT_PERM, act_perm)
    else:
        act_perm = read_file(ACT_PERM)

    if not STE_PERM.is_file():
        ste_perm = ste_matching()
        write_file(STE_PERM, ste_perm)
    else:
        ste_perm = read_file(STE_PERM)
    
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)
    mlp_model2.eval()

    results_dict = generate_plots(
        model1=mlp_model1,
        model2=mlp_model2,
        act_perm=act_perm,
        weight_perm=weight_perm,
        ste_perm=ste_perm,
    )

    # Creating a plot
    plt_dict(results_dict)
    write_file(MLP_RESULTS_PATH, results_dict)
