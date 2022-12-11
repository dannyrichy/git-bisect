from experiments.mlp_abalation import mlp_time_ablation,mlp_width_ablation, mlp_act_time_ablation, mlp_act_batch_ablation, mlp_width_act_ablation
from matplotlib import pyplot as plt
# from permuter.mlp import run
from permuter.vgg import run
# from models.vgg import train
# from torchvision.models import vgg16_bn
# from models.utils import cifar10_loader
from pathlib import Path

import torch
from torch import nn

from config import DEVICE, VGG_MODEL1_PATH, VGG_MODEL2_PATH, VGG_PERM_PATH
from helper import read_file
from models.utils import cifar10_loader
from models.vgg import train, vgg16_bn, vgg16_ln
from permuter.vgg import permute_model, run

if __name__ == "__main__":

    # train_loader, val_loader, test_loader = cifar10_loader(batch_size=256, validation=True, augument=True)

    # model = vgg16_bn(num_classes=10)
    # train(train_loader, val_loader, model, epochs=100, model_name="vgg")
    # mlp_width_ablation()
    
    
    # mlp_time_ablation()
    # mlp_act_batch_ablation()
    # mlp_width_act_ablation()
    # mlp_act_time_ablation()


    # train_loader, val_loader, test_loader = cifar10_loader(
    #     batch_size=256, validation=True, augument=True
    # )

    # model = vgg16_bn(num_classes=10)
    # train(train_loader, val_loader, model, epochs=100, model_name="vgg16_bn_100")

    # train_loader, val_loader, test_loader = cifar10_loader(
    #     batch_size=256, validation=True, augument=True
    # )

    # model = vgg16_bn(num_classes=10)
    # train(train_loader, val_loader, model, epochs=100, model_name="vgg16_bn1")

    run()
    # def weight_cost(model1: nn.Module, permuted_model2: nn.Module):
    #     cost = 0.0
    #     for param1, param2 in zip(model1.parameters(), permuted_model2.parameters()):
    #         cost += torch.sum(torch.mul(param1, param2)).item()
    #     return cost

    # vgg_model1, vgg_model2 = vgg16_ln(num_classes=10), vgg16_ln(num_classes=10)

    # vgg_model1.load_state_dict(torch.load(VGG_MODEL1_PATH))
    # vgg_model1.to(DEVICE)
    # vgg_model1.eval()

    # vgg_model2.load_state_dict(torch.load(VGG_MODEL2_PATH))
    # vgg_model2.to(DEVICE)
    # vgg_model2.eval()

    # WEIGHT_PERM = VGG_PERM_PATH.joinpath("weight_perm.pkl")
    # ACT_PERM = VGG_PERM_PATH.joinpath("act_perm.pkl")
    # act_perm = read_file(ACT_PERM)
    # weight_perm = read_file(WEIGHT_PERM)

    # act_perm_model2 = permute_model(model=vgg_model2, perm_dict=act_perm)
    # weight_perm_model2 = permute_model(model=vgg_model2, perm_dict=weight_perm)

    # print(
    #     f"weight cost using act perm: {weight_cost(model1=vgg_model1, permuted_model2=act_perm_model2):.3f}"
    # )
    # print(
    #     f"weight cost using weight perm: {weight_cost(model1=vgg_model1, permuted_model2=weight_perm_model2):.3f}"
    # )
