import time
from pathlib import Path

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
Utilities to help with creating hook
"""


"""
Training
"""


def cifar10_loader(batch_size:int)-> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:  # type: ignore
    """
    _summary_

    :return: _description_
    :rtype: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, # type: ignore
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,  # type: ignore
                                             shuffle=False)

    return trainloader, testloader

def train(trainloader, model, epochs, model_name='mlp'):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
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
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    print("Training done! 🤖")

    path = Path('./stash')
    path.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), path.joinpath(f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}.pth'))
    
    return model


def hook_func(res_dict: dict, name:str, module:torch.nn.modules.Module, inp: torch.Tensor, out:torch.Tensor) -> None:
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
    