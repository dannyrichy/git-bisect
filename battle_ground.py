from models import MLP
from models.utils import cifar10_loader, train

if  __name__ == '__main__':
    trainloader, testloader = cifar10_loader()

    mlp = MLP()
    mlp = train(trainloader, model=mlp, epochs=5)