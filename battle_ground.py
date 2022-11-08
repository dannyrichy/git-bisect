from functools import reduce

import numpy as np
import torch

from config import DEVICE, MLP_MODEL1_PATH, MLP_MODEL2_PATH
from core import ActivationMethod, LossBarrier
from models import MLP, cifar10_loader, register_hook

if __name__ == "__main__":
    train_loader, test_loader = cifar10_loader(batch_size=8)

    # TODO: Create checker methods using arch and model_width params
    permuter = ActivationMethod(arch=[512, 512, 512, 10], model_width=4)

    # Loading individually trained models
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.eval()

    model1_dict, model2_dict = dict(), dict()
    register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
    register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

    # TODO: Time the below two methods and get error value
    # Method 1: Evaluating cost matrix batch wise, values are
    # added element wise
    for inp, lbl in train_loader:
        _ = mlp_model1(inp)
        _ = mlp_model2(inp)

        # The dictionaries gets erased and updated every time
        permuter.evaluate_cost_batch_wise(model1_dict, model2_dict)

    # Fetching the permutation
    permutation_dict = permuter.get_permutation()

    # Creating loss_barrier loss function using the above permutation
    # matrix
    lb = LossBarrier(
        model1=mlp_model1.to(DEVICE),
        model2=mlp_model2.to(DEVICE),
        lambda_list=np.linspace(0, 1, 10),
        perm_dict=permutation_dict,
    )

    res = lb.loss_barrier(cifar10_loader(batch_size=128)[0].to(DEVICE))

    print("Done!")
