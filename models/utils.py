from typing import Union

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def cifar10_loader(
    batch_size: int, validation: bool = False
) -> Union[
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader],  # type: ignore
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],  # type: ignore
]:
    """
    _summary_

    :return: _description_
    :rtype: Union[
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader],
    tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_set, batch_size=batch_size, shuffle=False
    )

    if validation:
        train_set, val_set = random_split(train_set, [45000, 5000])
        val_loader = torch.utils.data.DataLoader(  # type: ignore
            val_set, batch_size=batch_size, shuffle=True
        )

        train_loader = torch.utils.data.DataLoader(  # type: ignore
            train_set, batch_size=batch_size, shuffle=True
        )
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(  # type: ignore
            train_set, batch_size=batch_size, shuffle=True
        )
        return train_loader, test_loader


def hook_func(
    res_dict: dict,
    name: str,
    module: torch.nn.modules.Module,
    inp: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Recipe for hook function, ensure to call partial on this
    with dictionary object to store the values

    :param res_dict: _description_
    :type res_dict: dict
    :param name:
    :type name: str
    :param module: _description_
    :type module: torch.nn.modules.Module
    :param inp: _description_
    :type inp: torch.Tensor
    :param out: _description_
    :type out: torch.Tensor
    """
    res_dict[name] = out
