from models.mlp import train
from models.mlp import MLP
from models import cifar10_loader
from models.utils import mnist_loader
from models.vgg import train as train_vgg
from models.vgg import vgg16_bn
import sys

if __name__ == "__main__":
    w = int(sys.argv[1])
    name = str(sys.argv[2])
    
    train_loader, validation_loader, test_loader = cifar10_loader(batch_size=512,validation=True, augument=True)
    # for w in [64,128,512,768]:
    print(f"Model with {w} training!")
    train(train_loader=train_loader, val_loader=validation_loader, model=MLP(WIDTH=w),epochs=40, model_name=f"{name}_{w}")
        
    