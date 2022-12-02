from pathlib import Path
from config import _STASH_PATH, DEVICE, MLP_PERM_PATH, NAIVE_MATCH, TEST, TRAIN, WEIGHT_MATCH
import torch
from helper import plt_dict, write_file
from permuter._algo import WeightMatching

from models.mlp import LAYER_NAMES, MLP, WEIGHT_PERM_LOOKUP
from permuter.mlp import generate_plots
from matplotlib import pyplot as plt
# from permuter.vgg import run
# from models.vgg import train
# from torchvision.models import vgg16_bn
# from models.utils import cifar10_loader

def mlp_width_ablation():
    """
    weight matching

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    abalation_path = MLP_PERM_PATH.joinpath("width_abalation")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "width":list(),
        TRAIN:list(),
        TEST:list(),
    }
    for w in [64,128,256,512,1024]:
        print(f"Running for width {w}")
        mlp_model1, mlp_model2 = MLP(WIDTH=w), MLP(WIDTH=w)
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp_{w}_20.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_{w}_20.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()

        weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
        _permutation_dict = weight_matcher.evaluate_permutation(
            m1_weights=mlp_model1.state_dict(), m2_weights=mlp_model2.state_dict()
        )
        
        write_file(abalation_path.joinpath(f"{str(w)}.pkl"),_permutation_dict)
        results_dict = generate_plots(
            model1=mlp_model1,
            model2=mlp_model2,
            weight_perm=_permutation_dict,
            width=w
        )
        plt_dict(results=results_dict, name=f"mlp_width_{str(w)}")
        loss_barrier["width"].append(w)
        loss_barrier[TRAIN].append(
            max(results_dict[WEIGHT_MATCH][TRAIN]) - 0.5*(results_dict[WEIGHT_MATCH][TRAIN][0]+results_dict[WEIGHT_MATCH][TRAIN][-1])
        )
           
        loss_barrier[TEST].append(
            max(results_dict[WEIGHT_MATCH][TEST]) - 0.5*(results_dict[WEIGHT_MATCH][TEST][0]+results_dict[WEIGHT_MATCH][TEST][-1])
        )
        
    write_file(Path("loss_barrier_mlp_width.pkl"), loss_barrier)
    plt.figure()
    _fmt = {
        TRAIN: {"linestyle": "solid", "marker": "*"},
        TEST: {"linestyle": "dashed", "marker": "*"}
        }
    plt.plot(loss_barrier["width"], loss_barrier[TRAIN], label="Train", **_fmt[TRAIN])
    plt.plot(loss_barrier["width"], loss_barrier[TEST], label="Test", **_fmt[TEST])

    plt.xlabel("Width")
    plt.ylabel("Loss Barrier")
    plt.legend()
    plt.savefig(f"Width_LossBarrier_mlp")


def mlp_time_ablation():
    abalation_path = MLP_PERM_PATH.joinpath("width_abalation")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "time":list(),
        TRAIN:list(),
        TEST:list(),
    }
    for t in [1,4,9,19,39]:
        print(f"Running for epoch time {t+1}")
        mlp_model1, mlp_model2 = MLP(), MLP()
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp_512_{t}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_512_{t}_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()

        weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
        _permutation_dict = weight_matcher.evaluate_permutation(
            m1_weights=mlp_model1.state_dict(), m2_weights=mlp_model2.state_dict()
        )
        
        write_file(abalation_path.joinpath(f"{str(t)}.pkl"),_permutation_dict)
        results_dict = generate_plots(
            model1=mlp_model1,
            model2=mlp_model2,
            weight_perm=_permutation_dict,
        )
        plt_dict(results=results_dict, name=f"mlp_epoch_{str(t)}")
        loss_barrier["time"].append(t)
        loss_barrier[TRAIN].append(
            max(results_dict[WEIGHT_MATCH][TRAIN]) - 0.5*(results_dict[WEIGHT_MATCH][TRAIN][0]+results_dict[WEIGHT_MATCH][TRAIN][-1])
        )
           
        loss_barrier[TEST].append(
            max(results_dict[WEIGHT_MATCH][TEST]) - 0.5*(results_dict[WEIGHT_MATCH][TEST][0]+results_dict[WEIGHT_MATCH][TEST][-1])
        )
        
    write_file(Path("loss_barrier_mlp_epoch.pkl"), loss_barrier)
    plt.figure()
    _fmt = {
        TRAIN: {"linestyle": "solid", "marker": "*"},
        TEST: {"linestyle": "dashed", "marker": "*"}
        }
    plt.plot(loss_barrier["time"], loss_barrier[TRAIN], label="Train", **_fmt[TRAIN])
    plt.plot(loss_barrier["time"], loss_barrier[TEST], label="Test", **_fmt[TEST])

    plt.xlabel("Epoch time")
    plt.ylabel("Loss Barrier")
    plt.legend()
    plt.savefig(f"Time_LossBarrier_mlp")
    

if __name__ == "__main__":

    # train_loader, val_loader, test_loader = cifar10_loader(batch_size=256, validation=True, augument=True)

    # model = vgg16_bn(num_classes=10)
    # train(train_loader, val_loader, model, epochs=100, model_name="vgg")
    # mlp_width_ablation()
    mlp_time_ablation()
