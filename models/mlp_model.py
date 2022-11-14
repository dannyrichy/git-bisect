import time
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn

from config import DEVICE
from models.utils import hook_func


class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(32 * 32 * 3, 512)
        self.relu_layer_1 = nn.ReLU()
        self.layer_2 = nn.Linear(512, 512)
        self.relu_layer_2 = nn.ReLU()
        self.layer_3 = nn.Linear(512, 512)
        self.relu_layer_3 = nn.ReLU()
        self.layer_4 = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor):
        """
        _summary_

        :param x: Input variable
        :type x: torch.Tensor

        :return: predicted output
        :rtype: torch.Tensor
        """
        y = torch.flatten(x, 1)
        y = self.layer_1(y)
        y = self.relu_layer_1(y)
        y = self.layer_2(y)
        y = self.relu_layer_2(y)
        y = self.layer_3(y)
        y = self.relu_layer_3(y)
        y = self.layer_4(y)

        return y


def register_hook(mlp_inst: MLP, activations_dict: dict) -> None:
    """
    Function to register hook

    :param mlp_inst: _description_
    :type mlp_inst: MLP
    :param activations_dict: _description_
    :type activations_dict: dict
    """
    for name, layer in mlp_inst.named_modules():
        if name.startswith("layer"):
            layer.register_forward_hook(hook=partial(hook_func, activations_dict, name))


def mlp_train(
    train_loader: torch.utils.data.DataLoader, # type: ignore
    val_loader: torch.utils.data.DataLoader,  # type: ignore
    model: MLP,
    epochs: int,
    model_name: str = "mlp",
) -> MLP:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.to(DEVICE)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs.to(DEVICE))
            loss = criterion(outputs, labels.to(DEVICE))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                val_loss = 0.0
                corr = 0.0
                with torch.no_grad():
                    for j, (inputs, labels) in enumerate(val_loader):
                        labels = labels.to(DEVICE)
                        outputs = model(inputs.to(DEVICE))
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, dim=1)
                        corr += torch.sum(preds == labels).item()
                    accuracy = corr / len(val_loader.dataset)
                print(
                    f"[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 2000:.3f} val_loss: {val_loss / (j+1):.3f} accuracy: {accuracy*100:.3f}%"
                )
                running_loss = 0.0
    print("Training done! 🤖")

    path = Path("./stash")
    path.mkdir(exist_ok=True, parents=True)
    torch.save(
        model.state_dict(),
        path.joinpath(f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}.pth'),
    )

    return model
