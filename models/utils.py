import time
from pathlib import Path

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models.mlp_model import MLP

"""
Utilities to help with creating hook
"""


"""
Training
"""


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


def train(
    train_loader: torch.utils.data.DataLoader,
    model: MLP,
    epochs: int,
    model_name: str = "mlp",
) -> MLP:
    criterion = torch.nn.CrossEntropyLoss()
    # @Adhithyan8 TODO: #5 Check if it is passed as function or function name
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    print("Training done! 🤖")

    path = Path("./stash")
    path.mkdir(exist_ok=True, parents=True)
    torch.save(
        model.state_dict(),
        path.joinpath(f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}.pth'),
    )

    return model


def hook_func(
    res_dict: dict,
    name: str,
    module: torch.nn.modules.Module,
    inp: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """
    Reciepe for hook function, ensure to call partial on this
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
