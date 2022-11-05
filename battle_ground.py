import torch
import numpy as np

from config import MLP_MODEL1_PATH, MLP_MODEL2_PATH
from core.algo import ActivationMethod
from models.mlp_model import MLP, register_hook
from models.utils import cifar10_loader, train

if  __name__ == '__main__':
    trainloader, testloader = cifar10_loader(batch_size=8)
    # mlp = train(trainloader, model=mlp, epochs=5)
    
    permuter = ActivationMethod(archi=[512,512, 512, 10], model_width=None)
    
    mlp_model1, mlp_model2 = MLP(), MLP()
    mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
    mlp_model1.eval()

    mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
    mlp_model2.eval()


    model1_dict, model2_dict = dict(), dict()
    register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
    register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

    hist_perm = None
    cost = list()
    
    for inp, lbl in trainloader:
        _ = mlp_model1(inp)
        _ = mlp_model2(inp)
        if hist_perm is None:
            hist_perm = [np.zeros((i.shape[1], i.shape[1]), dtype=np.float64) for i in model1_dict.values()]
        perm = permuter.get_permuation(model1_dict,  model2_dict)
        cost.append(permuter.get_loss())
        hist_perm = [i+j for i,j in zip(perm, hist_perm)]
        
    # Aggreement along row as categorical distribution
    for p in hist_perm:
        print(np.max(p, axis=1)/6250)         
    