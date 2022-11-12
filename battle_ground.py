from typing import Optional

import numpy as np
import torch

from config import (
    ACT_PERM,
    DEVICE,
    LAMBDA_ARRAY,
    MLP_MODEL1_PATH,
    MLP_MODEL2_PATH,
    MLP_PERM_PATH,
    WEIGHT_PERM,
)
from core import (
    ActMatching,
    WeightMatching,
    combine_models,
    loss_barrier,
    permute_model,
)
from helper import read_file, write_file
from models import MLP, cifar10_loader, register_hook


def activation_matching() -> dict[str, torch.Tensor]:
    train_loader, test_loader = cifar10_loader(batch_size=8)
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

    train_loader, test_loader = cifar10_loader(batch_size=128)
    result: dict[str, dict[str, np.ndarray]] = dict()

    def _generate_models(_model2: torch.nn.Module) -> dict[str, np.ndarray]:
        _models = list()
        for lam in LAMBDA_ARRAY:
            tmp = combine_models(model1=model1, model2=_model2, lam=lam)
            tmp.eval()
            _models.append(tmp)
        _res = {
            "Train": loss_barrier(
                data_loader=train_loader,
                model1=model1,
                model2=model2,
                combined_models=_models,
            ),
            "Test": loss_barrier(
                data_loader=test_loader,
                model1=model1,
                model2=model2,
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


if __name__ == "__main__":
    # res = activation_matching()

    if not WEIGHT_PERM.is_file():
        weight_perm = weight_matching()
        write_file(WEIGHT_PERM, weight_perm)
    else:
        weight_perm = read_file(WEIGHT_PERM)

    if not ACT_PERM.is_file():
        act_perm = activation_matching()
        write_file(MLP_PERM_PATH.joinpath("act_perm.pkl"), act_perm)
    else:
        act_perm = read_file(ACT_PERM)

    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.to(DEVICE)
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.to(DEVICE)
    mlp_model2.eval()

    results_dict = generate_plots(
        model1=mlp_model1, model2=mlp_model2, act_perm=act_perm, weight_perm=weight_perm
    )
