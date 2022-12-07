import time
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

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

INDEX_LAYER = {
    "features.1": ("features.0", "features.3"),
    "features.4": ("features.3", "features.7"),
    "features.8": ("features.7", "features.10"),
    "features.11": ("features.10", "features.14"),
    "features.15": ("features.14", "features.17"),
    "features.18": ("features.17", "features.20"),
    "features.21": ("features.20", "features.24"),
    "features.25": ("features.24", "features.27"),
    "features.28": ("features.27", "features.30"),
    "features.31": ("features.30", "features.34"),
    "features.35": ("features.34", "features.37"),
    "features.38": ("features.37", "features.40"),
    "features.41": ("features.40", "classifier.0"),
    "classifier.0": ("classifier.3"),
    "classifier.3": ("classifier.6"),
}

WEIGHT_PERM_LOOKUP = {
    "features.1": (None, "features.0", "features.3", "features.4"),
    "features.4": ("features.1", "features.3", "features.7", "features.8"),
    "features.8": ("features.4", "features.7", "features.10", "features.11"),
    "features.11": ("features.8", "features.10", "features.14", "features.15"),
    "features.15": ("features.11", "features.14", "features.17", "features.18"),
    "features.18": ("features.15", "features.17", "features.20", "features.21"),
    "features.21": ("features.18", "features.20", "features.24", "features.25"),
    "features.25": ("features.21", "features.24", "features.27", "features.28"),
    "features.28": ("features.25", "features.27", "features.30", "features.31"),
    "features.31": ("features.28", "features.30", "features.34", "features.35"),
    "features.35": ("features.31", "features.34", "features.37", "features.38"),
    "features.38": ("features.35", "features.37", "features.40", "features.41"),
    "features.41": ("features.38", "features.40", "classifier.0", "classifier.0"),
    "classifier.0": ("features.41", "classifier.3"),
    "classifier.3": ("classifier.0", "classifier.6"),
}

LOOK_UP_WEIGHTS = [
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
]

# define custom VGG with batch norm and layernorm
class VGG16_BN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        init_weights: bool = True,
        dropout: float = 0.5,
        multiplier: int = 64,
        in_channels:int =3
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 1 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(1 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(1 * multiplier, 1 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(1 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1 * multiplier, 2 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * multiplier, 2 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2 * multiplier, 4 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * multiplier, 4 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * multiplier, 4 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(4 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.BatchNorm2d(8 * multiplier),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(8 * multiplier, 8 * multiplier),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(8 * multiplier, 8 * multiplier),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(8 * multiplier, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16_bn(**kwargs: Any) -> VGG16_BN:
    return VGG16_BN(**kwargs)


class VGG16_LN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        init_weights: bool = True,
        dropout: float = 0.5,
        multiplier: int = 64,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 1 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(inplace=True),
            nn.Conv2d(1 * multiplier, 1 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([64, 32, 32]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1 * multiplier, 2 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * multiplier, 2 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([128, 16, 16]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2 * multiplier, 4 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * multiplier, 4 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * multiplier, 4 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([256, 8, 8]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([512, 4, 4]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(8 * multiplier, 8 * multiplier, kernel_size=3, padding=1),
            nn.LayerNorm([512, 2, 2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(8 * multiplier, 8 * multiplier),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(8 * multiplier, 8 * multiplier),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(8 * multiplier, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16_ln(**kwargs: Any) -> VGG16_LN:
    return VGG16_LN(**kwargs)


def register_hook(inst: VGG16_LN, activations_dict: dict) -> None:
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
            "weight_decay": 1e-4,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in LOOK_UP_WEIGHTS)
            ],
            "weight_decay": 0.0,
        },
    ]

    criterion = cross_entropy
    optimizer = optim.SGD(optimizer_parameters, lr=1e-5, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=epochs
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
                    accuracy = corr / len(val_loader.dataset)  # type:ignore
                print(
                    f"[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 2000:.3f} val_loss: {val_loss /len(val_loader.dataset):.3f} accuracy: {accuracy*100:.3f}%"  # type:ignore
                )
                running_loss = 0.0
            scheduler.step()
    print("Training done! 🤖")

    path = Path("./stash")
    path.mkdir(exist_ok=True, parents=True)
    torch.save(
        model.state_dict(),
        path.joinpath(f'{model_name}_{epochs}.pth'),
    )

    return model
