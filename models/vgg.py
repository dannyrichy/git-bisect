from functools import partial

from torchvision.models import vgg16_bn

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
    "classifier.6"
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
