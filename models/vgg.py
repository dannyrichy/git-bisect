from functools import partial

from torchvision.models import vgg16_bn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time

from config import DEVICE

from models.utils import hook_func

LOOK_UP_LAYER = {
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
}


def register_hook(mlp_inst: vgg16_bn, activations_dict: dict) -> None:
    """
    Function to register hook

    :param mlp_inst: _description_
    :type mlp_inst: MLP
    :param activations_dict: _description_
    :type activations_dict: dict
    """
    for name, module_par in mlp_inst.named_modules():
        for child_name, child_module in module_par.named_modules():
            tmp = name + "." + child_name
            if tmp in LOOK_UP_LAYER:
                module_par.register_forward_hook(
                    hook=partial(hook_func, activations_dict, tmp)
                )


# loads model with random weights (DELETE)


def vgg_train(
    train_loader,
    val_loader,
    model,
    epochs,
    model_name="vgg",
):
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
                    for inputs, labels in val_loader:
                        labels = labels.to(DEVICE)
                        outputs = model(inputs.to(DEVICE))
                        loss = criterion(outputs, labels, reduction="sum")
                        val_loss += loss.item()
                        _, preds = torch.max(outputs, dim=1)
                        corr += torch.sum(preds == labels).item()
                    accuracy = corr / len(val_loader.dataset)
                print(
                    f"[{epoch + 1}, {i + 1:5d}] train_loss: {running_loss / 2000:.3f} val_loss: {val_loss / len(val_loader.dataset):.3f} accuracy: {accuracy*100:.3f}%"
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
