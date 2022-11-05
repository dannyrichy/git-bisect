import numpy as np

from core.algo import ActivationMethod
from models.helper import fetch_activations
from models.utils import cifar10_loader, train

if  __name__ == '__main__':
    trainloader, testloader = cifar10_loader()
    # mlp = train(trainloader, model=mlp, epochs=5)
    
    act1, act2 = fetch_activations(trainloader)
    permuter = ActivationMethod(archi=[512,512, 512, 10], model_width=None)
    perm = permuter.get_permuation(act1,  act2)
    
    print(perm)            
    