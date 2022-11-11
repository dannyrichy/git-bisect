from functools import reduce
from typing import Optional

import numpy as np
import torch

from config import DEVICE, MLP_MODEL1_PATH, MLP_MODEL2_PATH
from core import (
    ActMatching,
    WeightMatching,
    combine_models,
    loss_barrier,
    permute_model,
)
from models import MLP, cifar10_loader, mlp_model, register_hook


def activation_matching() -> dict[str, torch.Tensor]:
    train_loader, test_loader = cifar10_loader(batch_size=8)
    # TODO: Create checker methods using arch and model_width params
    permuter = ActMatching(arch=[512, 512, 512, 10])

    # Loading individually trained models
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.eval()

    model1_dict, model2_dict = dict(), dict()
    register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
    register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

    # TODO: Time the below two methods and get error value
    # Method 1: Evaluating cost matrix batch wise, values are
    # added element wise
    for inp, lbl in train_loader:
        _ = mlp_model1(inp)
        _ = mlp_model2(inp)

        # The dictionaries gets erased and updated every time
        permuter.evaluate_cost_batch_wise(model1_dict, model2_dict)

    # Fetching the permutation
    permutation_dict = permuter.get_permutation()

    return {key: torch.Tensor(val).to(DEVICE) for key, val in permutation_dict.items()}


def weight_matching() -> dict[str, torch.Tensor]:
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
):
    # Creating loss_barrier loss function using the above permutation
    # matrix
    lambda_list = np.linspace(0, 1, 11)

    train_loader, test_loader = cifar10_loader(batch_size=128)
    result = dict()

    def _generate_models(_model2: torch.nn.Module) -> dict[str, np.ndarray]:
        _models = list()
        for lam in lambda_list:
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
    weight_perm = weight_matching()
    act_perm = activation_matching()

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
