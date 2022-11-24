import time
from functools import partial
from pathlib import Path

import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
from torchvision.models import VGG

from config import DEVICE
from models.utils import hook_func

LOOK_UP_LAYER = [
    "features.1",
    "features.4",
    "features.8",
    "features.11",
    "features.15",
    "features.18",
    "features.21",
    "features.25",
    "features.28",
    "features.31",
    "features.35",
    "features.38",
    "features.41",
    "classifier.0",
    "classifier.3",
    "classifier.6",
]

LOOK_UP_WEIGHTS = {
    "features.0.weight",
    "features.3.weight",
    "features.7.weight",
    "features.10.weight",
    "features.14.weight",
    "features.17.weight",
    "features.20.weight",
    "features.24.weight",
    "features.27.weight",
    "features.30.weight",
    "features.34.weight",
    "features.37.weight",
    "features.40.weight",
    "classifier.0.weight",
    "classifier.3.weight",
    "classifier.6.weight",
}


def register_hook(inst: VGG, activations_dict: dict) -> None:
    """
    Function to register hook

    :param inst: _description_
    :type inst: MLP
    :param activations_dict: _description_
    :type activations_dict: dict
    """
    for name, module_par in inst.named_modules():
        for child_name, child_module in module_par.named_modules():
            tmp = name + "." + child_name
            if tmp in LOOK_UP_LAYER:
                child_module.register_forward_hook(
                    hook=partial(hook_func, activations_dict, tmp)
                )


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: torch.nn.Module,
    epochs: int,
    model_name: str = "vgg",
):
    optimizer_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in LOOK_UP_WEIGHTS)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in LOOK_UP_WEIGHTS)
            ],
            "weight_decay": 0.0,
        },
    ]

    criterion = cross_entropy
    optimizer = optim.SGD(optimizer_parameters, lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=epochs
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
            if i % 30 == 29:
                val_loss = 0.0
                corr = 0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        labels = labels.to(DEVICE)
                        outputs = model(inputs.to(DEVICE))
                        loss = cross_entropy(outputs, labels, reduction="sum")
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, dim=1)
                        corr += torch.sum(preds == labels).item()
                    accuracy = corr / len(val_loader.dataset)
                print(
                    f"[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 2000:.3f} val_loss: {val_loss /len(val_loader.dataset):.3f} accuracy: {accuracy*100:.3f}%"
                )
                running_loss = 0.0
            scheduler.step()
    print("Training done! 🤖")

    path = Path("./stash")
    path.mkdir(exist_ok=True, parents=True)
    torch.save(
        model.state_dict(),
        path.joinpath(f'{model_name}_{time.strftime("%Y%m%d-%H%M%S")}.pth'),
    )

    return model
