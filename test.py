# File for testing algorithms

import torch

from config import MLP_MODEL1_PATH, MLP_MODEL2_PATH
from core.algo import ActMatching
from models.mlp import MLP, register_hook
from models.utils import cifar10_loader

trainloader, testloader = cifar10_loader(8)
# mlp = train(trainloader, model=mlp, epochs=5)

permuter = ActMatching(arch=[512, 512, 512, 10], model_width=None)

mlp_model1, mlp_model2 = MLP(), MLP()
mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
mlp_model1.eval()

mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
mlp_model2.eval()


model1_dict, model2_dict = dict(), dict()
register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

hist_perm = None
act_perm = list()
cost = list()

for inp, lbl in testloader:
    _ = mlp_model1(inp)
    if hist_perm is None:
        hist_perm = list()
        act_perm = list()
        for i in model1_dict.values():
            hist_perm.append(torch.eye(i.shape[1], dtype=torch.float64))
            act_perm.append(
                torch.eye(i.shape[1], dtype=torch.float64)[torch.randperm(i.shape[1])]
            )

    tmp = {
        key: torch.matmul(act_perm[ix], value.T.type(torch.DoubleTensor)).T
        for ix, (key, value) in enumerate(model1_dict.items())
    }
    tmp = {
        key: torch.matmul(hist_perm[ix], value.T).T
        for ix, (key, value) in enumerate(tmp.items())
    }
    perm = permuter.get_permutation(model1_dict, tmp)
    cost.append(permuter.get_loss())
    hist_perm = [torch.matmul(torch.from_numpy(i), j) for i, j in zip(perm, hist_perm)]

print(hist_perm)
