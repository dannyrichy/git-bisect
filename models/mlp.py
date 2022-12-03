import time
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from config import BIAS, DEVICE, WEIGHT
from models.utils import hook_func

LAYER_NAMES = ["layer_1", "layer_2", "layer_3", "layer_4"]
INDEX_LAYER = {"layer_1": "layer_2", "layer_2": "layer_3", "layer_3": "layer_4"}

WEIGHT_PERM_LOOKUP = {
    "layer_1": (None, "layer_2"),
    "layer_2": ("layer_1", "layer_3"),
    "layer_3": ("layer_2", "layer_4"),
}

class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self, WIDTH:int=512) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(32 * 32 * 3, WIDTH)
        self.relu_layer_1 = nn.ReLU()
        self.layer_2 = nn.Linear(WIDTH, WIDTH)
        self.relu_layer_2 = nn.ReLU()
        self.layer_3 = nn.Linear(WIDTH, WIDTH)
        self.relu_layer_3 = nn.ReLU()
        self.layer_4 = nn.Linear(WIDTH, 10)

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


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: MLP,
    epochs: int,
    model_name: str = "mlp",
) -> MLP:
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n.endswith(WEIGHT))
            ],
            "weight_decay": 0.005,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n.endswith(BIAS))
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.SGD(optimizer_parameters, lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epochs
    )
    path = Path("./stash")
    path.mkdir(exist_ok=True, parents=True)
    model.to(DEVICE)
    
    print("Saving model before training")
    torch.save(
                model.to(torch.device("cpu")).state_dict(),
                path.joinpath(f'{model_name}_0_1_{epochs}.pth'),
            )
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
            print(f"Saving model at {epoch+1} & batch {i+1}")
            torch.save(
                model.to(torch.device("cpu")).state_dict(),
                path.joinpath(f'{model_name}_{i+1}_{epoch+1}_{epochs}.pth'),
            )
            model.to(DEVICE) 
            if i % 30 == 29:
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
                    accuracy = corr / len(val_loader.dataset)  # type:ignore
                print(
                    f"[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 30:.3f} val_loss: {val_loss / (j+1):.3f} accuracy: {accuracy*100:.3f}%"  # type:ignore
                )
                running_loss = 0.0
            scheduler.step()
               
    print("Training done! 🤖")

    
    # torch.save(
    #     model.to(torch.device("cpu")).state_dict(),
    #     path.joinpath(f'{model_name}_{epochs}.pth'),
    # )

    return model
