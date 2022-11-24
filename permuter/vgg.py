from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn

from config import (
    BIAS,
    CLASSIFIER,
    DEVICE,
    FEATURES,
    LAMBDA_ARRAY,
    VGG_MODEL1_PATH,
    VGG_MODEL2_PATH,
    VGG_PERM_PATH,
    WEIGHT,
)
from helper import plt_dict, read_file, write_file
from models import cifar10_loader
from models.vgg import LOOK_UP_LAYER, register_hook, train
from permuter._algo import ActMatching, STEstimator, WeightMatching
from permuter.common import combine_models, get_losses

WEIGHT_PERM = VGG_PERM_PATH.joinpath("weight_perm.pkl")
ACT_PERM = VGG_PERM_PATH.joinpath("act_perm.pkl")


def get_indices(perm_mat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.argmax(perm_mat, dim=1), torch.argmax(perm_mat, dim=0)


def permute_model(
    model: torch.nn.Module,
    perm_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    """
    Permute the model with the dictionary. We are ignoring buffers, as we would re-initialise them after combining

    :param model: Model to be permuted
    :type model: torch.nn.Module
    :param perm_dict: Permutation dictionary
    :type perm_dict: dict[str, torch.Tensor]
    :return: Permuted model
    :rtype: torch.nn.Module
    """
    # Creating model instance to hold the permuted model
    permuted_model = vgg16_bn(num_classes=10).to(DEVICE)

    perm_state_dict = permuted_model.state_dict()
    model2_state_dict = model.state_dict()
    _prev_perm, hand_over = None, True
    for key in perm_dict.keys():
        if key.startswith(FEATURES):
            # Key normally accounts for batch-norm
            # perm_state_dict[key + "." + WEIGHT] = model2_state_dict[key + "." + WEIGHT][
            #     row_ind
            # ]
            # perm_state_dict[key + "." + BIAS] = model2_state_dict[key + "." + BIAS][
            #     row_ind
            # ]
            # perm_state_dict[_key_tmp + "." + BIAS] = model2_state_dict[
            #     _key_tmp + "." + BIAS
            # ][row_ind]
            # perm_state_dict[_key_tmp + "." + WEIGHT] = model2_state_dict[
            #     _key_tmp + "." + WEIGHT
            # ][row_ind, :, :, :]


            # Changing for conv filters
            _prev_key = key.split(".")
            _prev_key = ".".join([_prev_key[0], str(int(_prev_key[1]) - 1)])

            _iter = [i + "." + j for i in (key, _prev_key) for j in (WEIGHT, BIAS)]
            
            # Forward permutation
            for _key in _iter:
                perm_state_dict[_key] = torch.einsum(
                    "ij, j... -> i...",
                    perm_dict[key],
                    model2_state_dict[_key],
                )

            if _prev_perm is not None:
                # perm_state_dict[_key_tmp + "." + WEIGHT] = perm_state_dict[
                #     _key_tmp + "." + WEIGHT
                # ][:, _col_ind, :, :]
                perm_state_dict[_prev_key + "." + WEIGHT] = torch.einsum(
                    "jk..., ik -> ji...",
                    model2_state_dict[_prev_key + "." + WEIGHT],
                    _prev_perm,
                )

        elif key.startswith(CLASSIFIER) and _prev_perm is not None:
            # perm_state_dict[key + "." + BIAS] = (
            #     model2_state_dict[key + "." + BIAS][row_ind]
            #     if not key.endswith("6")
            #     else model2_state_dict[key + "." + BIAS]
            # )
            perm_state_dict[key + "." + BIAS] = (
                torch.einsum(
                    "ij, j... -> i...",
                    perm_dict[key],
                    model2_state_dict[key + "." + BIAS],
                )
                if not key.endswith("6")
                else model2_state_dict[key + "." + BIAS]
            )

            # perm_state_dict[key + "." + WEIGHT] = (
            #     model2_state_dict[key + "." + WEIGHT][row_ind, :]
            #     if not key.endswith("6")
            #     else model2_state_dict[key + "." + WEIGHT]
            # )
            perm_state_dict[key + "." + WEIGHT] = (
                torch.einsum(
                    "ij, j... -> i...",
                    perm_dict[key],
                    model2_state_dict[key + "." + WEIGHT],
                )
                if not key.endswith("6")
                else model2_state_dict[key + "." + WEIGHT]
            )

            if hand_over:
                _shape = int(
                    model2_state_dict[key + "." + WEIGHT].shape[1]
                    / _prev_perm.size(dim=0)
                )
                # _rolled_col_ind = torch.Tensor(
                #     [i * _shape + j for i in _col_ind for j in range(_shape)]
                # )
                # perm_state_dict[key + "." + WEIGHT] = perm_state_dict[
                #     key + "." + WEIGHT
                # ][:, _rolled_col_ind]
                perm_state_dict[key + "." + WEIGHT] = torch.einsum(
                    "jk..., ki -> ji...",
                    model2_state_dict[key + "." + WEIGHT],
                    torch.kron(_prev_perm.T.contiguous(), torch.eye(_shape).to(DEVICE)),
                )
                hand_over = False
            else:
                perm_state_dict[key + "." + WEIGHT] = torch.einsum(
                    "jk..., ik -> ji...",
                    model2_state_dict[key + "." + WEIGHT],
                    _prev_perm,
                )

        _prev_perm = perm_dict[key]
    permuted_model.load_state_dict(perm_state_dict)
    return permuted_model


def activation_matching() -> dict[str, torch.Tensor]:
    """
    Activation matching code for VGG

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    print("Computing using Activation Matching!")
    train_loader, test_loader, _ = cifar10_loader(batch_size=256)
    permuter = ActMatching(arch=LOOK_UP_LAYER)

    # Loading individually trained models
    vgg_model1, vgg_model2 = vgg16_bn(num_classes=10), vgg16_bn(num_classes=10)
    vgg_model1.load_state_dict(torch.load(VGG_MODEL1_PATH))
    vgg_model1.to(DEVICE)
    vgg_model1.eval()

    vgg_model2.load_state_dict(torch.load(VGG_MODEL2_PATH))
    vgg_model2.to(DEVICE)
    vgg_model2.eval()

    model1_dict, model2_dict = dict(), dict()
    register_hook(inst=vgg_model1, activations_dict=model1_dict)
    register_hook(inst=vgg_model2, activations_dict=model2_dict)

    # TODO: Time the below two methods and get error value
    # Method 1: Evaluating cost matrix batch wise, values are
    # added element wise
    for inp, lbl in train_loader:
        _ = vgg_model1(inp.to(DEVICE))
        _ = vgg_model2(inp.to(DEVICE))

        # The dictionaries gets erased and updated every time
        permuter.evaluate_permutation(model1_dict, model2_dict)

    # Fetching the permutation
    permutation_dict = permuter.get_permutation()

    return permutation_dict


# TODO:Yet to check the following function
def weight_matching() -> dict[str, torch.Tensor]:
    """
    weight matching

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    # Loading individually trained models
    vgg_model1, vgg_model2 = vgg16_bn(num_classes=10), vgg16_bn(num_classes=10)
    vgg_model1.load_state_dict(torch.load(VGG_MODEL1_PATH))
    vgg_model1.to(DEVICE)
    vgg_model1.eval()

    vgg_model2.load_state_dict(torch.load(VGG_MODEL2_PATH))
    vgg_model2.to(DEVICE)
    vgg_model2.eval()

    weight_matcher = WeightMatching(arch=LOOK_UP_LAYER)
    _permutation_dict = weight_matcher.evaluate_permutation(
        m1_weights=vgg_model1.state_dict(), m2_weights=vgg_model2.state_dict()
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
    result["NaiveMatching"] = _generate_models(_model2=model2)
    print("Done!")
    return result


def run():

    # train_loader, val_loader, test_loader = cifar10_loader(batch_size=256, validation=True, augument=True)

    # model = vgg16_bn(num_classes=10)
    # train(train_loader, val_loader, model, epochs=20, model_name="vgg")

    if not ACT_PERM.is_file():
        act_perm = activation_matching()
        write_file(ACT_PERM, act_perm)
    else:
        act_perm = read_file(ACT_PERM)

    vgg_model1, vgg_model2 = vgg16_bn(num_classes=10), vgg16_bn(num_classes=10)
    vgg_model1.load_state_dict(torch.load(VGG_MODEL1_PATH))
    vgg_model1.to(DEVICE)
    vgg_model1.eval()

    vgg_model2.load_state_dict(torch.load(VGG_MODEL2_PATH))
    vgg_model2.to(DEVICE)
    vgg_model2.eval()

    results_dict = generate_plots(
        model1=vgg_model1, model2=vgg_model2, act_perm=act_perm
    )
    plt_dict(results_dict)
