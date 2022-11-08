from functools import reduce

import numpy as np
import torch

from config import MLP_MODEL1_PATH, MLP_MODEL2_PATH
from core import ActivationMethod, loss_barrier
from models.mlp_model import MLP, register_hook
from models.utils import cifar10_loader, train

if __name__ == "__main__":
    train_loader, test_loader = cifar10_loader(batch_size=8)
    # mlp = train(train_loader, model=mlp, epochs=5)

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
    lb = loss_barrier(
        model1=mlp_model1,
        model2=mlp_model2,
        lambda_list=np.linspace(0, 1, 50),
        perm_dict=permutation_dict,
    )

    list_res = reduce(lambda x, y: x | y, [lb(inp, lbl) for inp, lbl in train_loader])
