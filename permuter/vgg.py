import copy
from typing import Optional

import numpy as np
import torch
from torchvision.models import vgg16_bn

from config import (
    ACT_MATCH,
    BIAS,
    CLASSIFIER,
    DEVICE,
    FEATURES,
    LAMBDA_ARRAY,
    NAIVE_MATCH,
    STE_MATCH,
    TEST,
    TRAIN,
    VGG_MODEL1_PATH,
    VGG_MODEL2_PATH,
    VGG_PERM_PATH,
    VGG_RESULTS_PATH,
    WEIGHT,
    WEIGHT_MATCH,
)
from helper import plt_dict, read_file, write_file
from models import cifar10_loader
from models.vgg import (
    INDEX_LAYER,
    LOOK_UP_LAYER,
    WEIGHT_PERM_LOOKUP,
    register_hook,
    train,
)
from permuter._algo import ActMatching, STEstimator, WeightMatching
from permuter.common import combine_models, get_losses

WEIGHT_PERM = VGG_PERM_PATH.joinpath("weight_perm.pkl")
ACT_PERM = VGG_PERM_PATH.joinpath("act_perm.pkl")


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

    perm_state_dict = copy.deepcopy(model.state_dict())
    model2_state_dict = model.state_dict()
    for key in perm_dict.keys():
        if key.startswith(FEATURES):
            # Key normally accounts for batch-norm

            # Changing for conv filters
            _prev_key, _next_key = INDEX_LAYER[key]
            _iter = [i + "." + j for i in (key, _prev_key) for j in (WEIGHT, BIAS)]

            # Forward permutation
            for _key in _iter:
                perm_state_dict[_key] = torch.einsum(
                    "ij, j... -> i...",
                    perm_dict[key],
                    perm_state_dict[_key],
                )

            # Reverse permutation for the next layer
            if not _next_key.startswith(CLASSIFIER):
                perm_state_dict[_next_key + "." + WEIGHT] = torch.einsum(
                    "jk..., ik -> ji...",
                    model2_state_dict[_next_key + "." + WEIGHT],
                    perm_dict[key],
                )
            else:
                _shape = int(
                    model2_state_dict[_next_key + "." + WEIGHT].shape[1]
                    / perm_dict[key].size(dim=0)
                )
                perm_state_dict[_next_key + "." + WEIGHT] = torch.einsum(
                    "jk..., ki -> ji...",
                    model2_state_dict[_next_key + "." + WEIGHT],
                    torch.kron(
                        perm_dict[key].T.contiguous(), torch.eye(_shape).to(DEVICE)
                    ),
                )

        elif key.startswith(CLASSIFIER):
            _next_key = INDEX_LAYER[key]
            _iter = [key + "." + j for j in (WEIGHT, BIAS)]
            for _key in _iter:
                perm_state_dict[_key] = torch.einsum(
                    "ij, j... -> i...",
                    perm_dict[key],
                    perm_state_dict[_key],
                )
            perm_state_dict[_next_key + "." + WEIGHT] = torch.einsum(
                "jk..., ik -> ji...",
                model2_state_dict[_next_key + "." + WEIGHT],
                perm_dict[key],
            )
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

    weight_matcher = WeightMatching(arch=LOOK_UP_LAYER, perm_lookup=WEIGHT_PERM_LOOKUP)
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
            torch.optim.swa_utils.update_bn(train_loader, tmp, device=DEVICE)
            _models.append(tmp)
        _res = {
            TRAIN: get_losses(
                data_loader=train_loader,
                combined_models=_models,
            ),
            TEST: get_losses(
                data_loader=test_loader,
                combined_models=_models,
            ),
        }
        return _res

    result[NAIVE_MATCH] = _generate_models(_model2=model2)
    if act_perm:
        _perm_model = permute_model(model=model2, perm_dict=act_perm)
        _perm_model.eval()
        result[ACT_MATCH] = _generate_models(_model2=_perm_model)
    if weight_perm:
        _perm_model = permute_model(model=model2, perm_dict=weight_perm)
        _perm_model.eval()
        result[WEIGHT_MATCH] = _generate_models(_model2=_perm_model)
    if ste_perm:
        _perm_model = permute_model(model=model2, perm_dict=ste_perm)
        _perm_model.eval()
        result[STE_MATCH] = _generate_models(_model2=_perm_model)
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
    write_file(VGG_RESULTS_PATH, results_dict)
