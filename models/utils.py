from typing import Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


def cifar10_loader(
    batch_size: int, validation: bool = False, augument: bool = False
) -> tuple[
    DataLoader[torchvision.datasets.CIFAR10],
    DataLoader[torchvision.datasets.CIFAR10],
    Optional[DataLoader[torchvision.datasets.CIFAR10]],
]:
    """
        _summary_

        :return: _description_
        :rtype: tuple[
        DataLoader[torchvision.datasets.CIFAR10],
        DataLoader[torchvision.datasets.CIFAR10],
        Optional[DataLoader[torchvision.datasets.CIFAR10]],
    ],
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    if augument:
        train_transform = transforms.Compose(
            [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=train_transform
        )
    else:
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    if validation:
        train_set, val_set = random_split(train_set, [45000, 5000])
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=True, num_workers=2
        )

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        return train_loader, test_loader, None


def mnist_loader(
    batch_size: int, validation: bool = False, augument: bool = False
) -> tuple[
    DataLoader[torchvision.datasets.CIFAR10],
    DataLoader[torchvision.datasets.CIFAR10],
    Optional[DataLoader[torchvision.datasets.CIFAR10]],
]:
    """
        _summary_

        :return: _description_
        :rtype: tuple[
        DataLoader[torchvision.datasets.CIFAR10],
        DataLoader[torchvision.datasets.CIFAR10],
        Optional[DataLoader[torchvision.datasets.CIFAR10]],
    ],
    """
    transform = transforms.Compose(
        [transforms.Resize(size=(32,32)), transforms.ToTensor(), transforms.Normalize((0.1307),
                    (0.3081) )]
    )

    if augument:
        train_transform = transforms.Compose(
            [transforms.Resize(size=(32,32)),   
             transforms.RandAugment(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307),
                    (0.3081)  
                ),
                    
            ]
        )

        train_set = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=train_transform
        )
    else:
        train_set = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )

    test_set = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    if validation:
        train_set, val_set = random_split(train_set, [50000, 10000])
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=True, num_workers=2
        )

        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        return train_loader, val_loader, test_loader
    else:
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=2
        )
        return train_loader, test_loader, None



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
    # Assuming the shape can either be of dimension 2 or 4
    res_dict[name] = out
