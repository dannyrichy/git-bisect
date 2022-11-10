import torch
import torchvision
import torchvision.transforms as transforms


def cifar10_loader(batch_size: int) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:  # type: ignore
    """
    _summary_

    :return: _description_
    :rtype: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_set, batch_size=batch_size, shuffle=True
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(  # type: ignore
        test_set, batch_size=batch_size, shuffle=False
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
