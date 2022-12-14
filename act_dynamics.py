from pprint import pprint
from config import  BIAS, DEVICE, WEIGHT
from models.mlp import INDEX_LAYER, LAYER_NAMES, MLP, WEIGHT_PERM_LOOKUP, register_hook
from models import cifar10_loader
import sys
import torch
import torch.optim as optim
from permuter._algo import ActMatching
from permuter.common import perm_linear_layer
from permuter.mlp import permute_model
import numpy as np

if __name__ == "__main__":
    w = 512
    
    train_loader, validation_loader, test_loader = cifar10_loader(batch_size=512,validation=True, augument=True)
    # for w in [64,128,512,768]:
    print(f"Model with {w} training!")
    
    model1=MLP(WIDTH=w).to(DEVICE)
    model2=MLP(WIDTH=w).to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_parameters1 = [
        {
            "params": [
                p for n, p in model1.named_parameters() if (n.endswith(WEIGHT))
            ],
            "weight_decay": 0.005,
        },
        {
            "params": [
                p for n, p in model1.named_parameters() if (n.endswith(BIAS))
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer_parameters2 = [
        {
            "params": [
                p for n, p in model2.named_parameters() if (n.endswith(WEIGHT))
            ],
            "weight_decay": 0.005,
        },
        {
            "params": [
                p for n, p in model2.named_parameters() if (n.endswith(BIAS))
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer1 = optim.SGD(optimizer_parameters1, lr=0.001, momentum=0.9)  
    optimizer2 = optim.SGD(optimizer_parameters2, lr=0.001, momentum=0.9)
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer1.zero_grad()
        # forward + backward + optimize
        outputs = model1(inputs.to(DEVICE))
        loss1 = criterion(outputs, labels.to(DEVICE))
        loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        # forward + backward + optimize
        outputs = model2(inputs.to(DEVICE))
        loss2 = criterion(outputs, labels.to(DEVICE))
        loss2.backward()
        optimizer2.step()
          
    
    
    permuter = ActMatching(arch=LAYER_NAMES)
    model1_dict, model2_dict = dict(), dict()
    register_hook(mlp_inst=model1, activations_dict=model1_dict)
    register_hook(mlp_inst=model2, activations_dict=model2_dict)

    # TODO: Time the below two methods and get error value
    # Method 1: Evaluating cost matrix batch wise, values are
    # added element wise
    for inp, lbl in train_loader:
        _ = model1(inp.to(DEVICE))
        _ = model2(inp.to(DEVICE))

        # The dictionaries gets erased and updated every time
        permuter.evaluate_permutation(model1_dict, model2_dict)

    # Fetching the permutation
    _permutation_dict = permuter.get_permutation()
    
    permuted_model = permute_model(model=model2, perm_dict=_permutation_dict, width=w)
    optimizer_parameters_perm = [
        {
            "params": [
                p for n, p in permuted_model.named_parameters() if (n.endswith(WEIGHT))
            ],
            "weight_decay": 0.005,
        },
        {
            "params": [
                p for n, p in permuted_model.named_parameters() if (n.endswith(BIAS))
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer_perm = optim.SGD(optimizer_parameters_perm, lr=0.001, momentum=0.9)
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer1.zero_grad()
        # forward + backward + optimize
        outputs = model1(inputs.to(DEVICE))
        loss1 = criterion(outputs, labels.to(DEVICE))
        loss1.backward()
        optimizer1.step()
        
        optimizer2.zero_grad()
        # forward + backward + optimize
        outputs = model2(inputs.to(DEVICE))
        loss2 = criterion(outputs, labels.to(DEVICE))
        loss2.backward()
        optimizer2.step()
        
        optimizer_perm.zero_grad()
        # forward + backward + optimize
        outputs = permuted_model(inputs.to(DEVICE))
        loss_perm = criterion(outputs, labels.to(DEVICE))
        loss_perm.backward()
        optimizer_perm.step()
    
    # Forgive me
    def _get_grad(model):
        return {
            "layer_1.weight": model.layer_1.weight.grad,
            "layer_1.bias": model.layer_1.bias.grad,
            "layer_2.weight": model.layer_2.weight.grad,
            "layer_2.bias": model.layer_2.bias.grad,
            "layer_3.weight": model.layer_3.weight.grad,
            "layer_3.bias": model.layer_3.bias.grad,
            "layer_4.weight": model.layer_4.weight.grad,
            "layer_4.bias": model.layer_4.bias.grad 
        }
    
    grad_model1 = _get_grad(model1)
    grad_model2 = _get_grad(model2)
    grad_perm_model = _get_grad(permuted_model)
    grad_model2_new = perm_linear_layer(
        model_sd=grad_model2, perm_dict=_permutation_dict, layer_look_up=INDEX_LAYER
    ) 
    
    
    # Checking if graients are  equal
    res = {
        k: np.sum(np.multiply(grad_model2_new[k].detach().cpu().numpy() ,grad_perm_model[k].detach().cpu().numpy()))/(np.linalg.norm(grad_model2_new[k].detach().cpu().numpy())*np.linalg.norm(grad_perm_model[k].detach().cpu().numpy()))
        for k in grad_model2_new.keys()
    }
    pprint(res)
    
    res = {
        k: np.linalg.norm(grad_model2_new[k].detach().cpu().numpy())/np.linalg.norm(grad_perm_model[k].detach().cpu().numpy())
        for k in grad_model2_new.keys()
    }
    pprint(res)
    
    
    permuter = ActMatching(arch=LAYER_NAMES)
    model1_dict, model2_dict = dict(), dict()
    register_hook(mlp_inst=model1, activations_dict=model1_dict)
    register_hook(mlp_inst=model2, activations_dict=model2_dict)

    # TODO: Time the below two methods and get error value
    # Method 1: Evaluating cost matrix batch wise, values are
    # added element wise
    for inp, lbl in train_loader:
        _ = model1(inp.to(DEVICE))
        _ = model2(inp.to(DEVICE))

        # The dictionaries gets erased and updated every time
        permuter.evaluate_permutation(model1_dict, model2_dict)

    # Fetching the permutation
    _permutation_dict2 = permuter.get_permutation()
    
    res = {
        k: np.sum(np.abs(_permutation_dict2[k].detach().cpu().numpy() - _permutation_dict[k].detach().cpu().numpy()))
        for k in _permutation_dict2.keys()
    }
    pprint(res)
    
    