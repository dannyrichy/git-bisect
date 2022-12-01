# from permuter.mlp import run
from permuter.vgg import run
from models.vgg import train
from torchvision.models import vgg16_bn
from models.utils import cifar10_loader

if __name__ == "__main__":

    train_loader, val_loader, test_loader = cifar10_loader(batch_size=256, validation=True, augument=True)

    model = vgg16_bn(num_classes=10)
    train(train_loader, val_loader, model, epochs=100, model_name="vgg")
    # run()
