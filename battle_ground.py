from functools import reduce

import numpy as np
import torch

from config import DEVICE, MLP_MODEL1_PATH, MLP_MODEL2_PATH
from core import ActMatching, combine_models, loss_barrier, permute_model
from models import MLP, cifar10_loader, register_hook

if __name__ == "__main__":
    train_loader, test_loader = cifar10_loader(batch_size=8)

    # TODO: Create checker methods using arch and model_width params
    permuter = ActMatching(arch=[512, 512, 512, 10])

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

    mlp_model1.to(DEVICE)
    mlp_model2.to(DEVICE)

    perm_model = permute_model(
        model=mlp_model2,
        perm_dict={
            key: torch.Tensor(val).to(DEVICE) for key, val in permutation_dict.items()
        },
    )
    perm_model.eval()
    lambda_list = np.arange(0, 1, 10)

    naive_models = list()
    for lam in lambda_list:
        tmp = combine_models(model1=mlp_model1, model2=mlp_model2, lam=lam)
        tmp.eval()
        naive_models.append(tmp)

    weight_matched_models = list()
    for lam in lambda_list:
        tmp = combine_models(model1=mlp_model1, model2=perm_model, lam=lam)
        tmp.eval()
        weight_matched_models.append(lam)

    res = (
        {
            "Naive combination": loss_barrier(
                data_loader=data_loader,
                model1=mlp_model1,
                model2=mlp_model2,
                combined_models=naive_models,
            ),
            "Weight Activation Matching": loss_barrier(
                data_loader=data_loader,
                model1=mlp_model1,
                model2=mlp_model2,
                combined_models=weight_matched_models,
            ),
        }
        for data_loader in cifar10_loader(batch_size=128)
    )

    print("Done!")
