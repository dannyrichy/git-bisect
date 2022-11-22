from typing import Optional
from torchvision.models import vgg16_bn

import numpy as np
import torch

from config import (
    DEVICE,
    LAMBDA_ARRAY,
    VGG_MODEL1_PATH,
    VGG_MODEL2_PATH,
)
from core import (
    ActMatching,
    STEstimator,
    WeightMatching,
    combine_models,
    get_losses,
    permute_model,
)
from models import  cifar10_loader
from models.vgg import register_hook, train


def activation_matching() -> dict[str, torch.Tensor]:
    """
    Activation matching code for VGG

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    train_loader, test_loader, _ = cifar10_loader(batch_size=256)
    # TODO: Create checker methods using arch and model_width params
    permuter = ActMatching(arch=[512, 512, 512, 10])

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

    weight_matcher = WeightMatching(arch=[512, 512, 512, 10])
    _permutation_dict = weight_matcher.evaluate_permutation(
        model1_weights=vgg_model1.state_dict(), model2_weights=vgg_model2.state_dict()
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


if __name__ == "__main__":

    # train_loader, val_loader, test_loader = cifar10_loader(batch_size=256, validation=True, augument=True)

    # model = vgg16_bn(num_classes=10)
    # train(train_loader, val_loader, model, epochs=20, model_name="vgg")
    activation_matching()