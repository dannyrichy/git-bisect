from pathlib import Path
from config import (
    _STASH_PATH,
    ACT_MATCH,
    DEVICE,
    LAMBDA_ARRAY,
    VGG_PERM_PATH,
    NAIVE_MATCH,
    TEST,
    TRAIN,
    WEIGHT_MATCH,
)
import torch
from helper import plt_dict, write_file
from models.utils import cifar10_loader
from permuter._algo import WeightMatching, ActMatching

from models.vgg import LOOK_UP_LAYER, WEIGHT_PERM_LOOKUP, register_hook, vgg16_bn
from permuter.vgg import generate_plots
from matplotlib import pyplot as plt

fmt = {
    TRAIN: {"linestyle": "solid", "marker": "*"},
    TEST: {"linestyle": "dashed", "marker": "*"},
    NAIVE_MATCH: {"color": "k"},
    ACT_MATCH: {"color": "r"},
    WEIGHT_MATCH: {"color": "g"},
}


def vgg_time_ablation():
    parent_folder = Path("vgg_epoch_plot")
    parent_folder.mkdir(exist_ok=True, parents=True)
    abalation_path = VGG_PERM_PATH.joinpath("time_abalation")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "time": list(),
        TRAIN: list(),
        TEST: list(),
    }

    for t in range(2, 101, 2):
        print(f"Running for epoch time {t}")
        vgg_model1, vgg_model2 = vgg16_bn(), vgg16_bn()
        vgg_model1.load_state_dict(
            torch.load(_STASH_PATH.joinpath(f"vgg16_bn1_{t}_100.pth"))
        )
        vgg_model1.to(DEVICE)
        vgg_model1.eval()

        vgg_model2.load_state_dict(
            torch.load(_STASH_PATH.joinpath(f"vgg16_bn2_{t}_100.pth"))
        )
        vgg_model2.to(DEVICE)
        vgg_model2.eval()

        weight_matcher = WeightMatching(
            arch=LOOK_UP_LAYER, perm_lookup=WEIGHT_PERM_LOOKUP
        )
        _permutation_dict = weight_matcher.evaluate_permutation(
            m1_weights=vgg_model1.state_dict(), m2_weights=vgg_model2.state_dict()
        )

        write_file(abalation_path.joinpath(f"{str(t)}.pkl"), _permutation_dict)
        results_dict = generate_plots(
            model1=vgg_model1,
            model2=vgg_model2,
            weight_perm=_permutation_dict,
        )
        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(
                    LAMBDA_ARRAY,
                    loss_arr,
                    label=method + "_" + set,
                    **fmt[set],
                    **fmt[method],
                )

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Weight match: Epoch {t}")
        plt.savefig(parent_folder.joinpath(f"vgg_epoch_{str(t)}"))
        plt.clf()
        loss_barrier["time"].append(t)
        loss_barrier[TRAIN].append(
            max(results_dict[WEIGHT_MATCH][TRAIN])
            - 0.5
            * (
                results_dict[WEIGHT_MATCH][TRAIN][0]
                + results_dict[WEIGHT_MATCH][TRAIN][-1]
            )
        )

        loss_barrier[TEST].append(
            max(results_dict[WEIGHT_MATCH][TEST])
            - 0.5
            * (
                results_dict[WEIGHT_MATCH][TEST][0]
                + results_dict[WEIGHT_MATCH][TEST][-1]
            )
        )

    write_file(Path("loss_barrier_vgg_epoch.pkl"), loss_barrier)
    plt.figure()
    _fmt = {
        TRAIN: {"linestyle": "solid", "marker": "*"},
        TEST: {"linestyle": "dashed", "marker": "*"},
    }
    plt.plot(loss_barrier["time"], loss_barrier[TRAIN], label="Train", **_fmt[TRAIN])
    plt.plot(loss_barrier["time"], loss_barrier[TEST], label="Test", **_fmt[TEST])

    plt.xlabel("Epoch time")
    plt.ylabel("Loss Barrier")
    plt.legend()
    plt.savefig(f"Time_LossBarrier_vgg")


def vgg_act_time_ablation():
    parent_folder = Path("vgg_epoch_act_plot")
    parent_folder.mkdir(exist_ok=True, parents=True)
    abalation_path = VGG_PERM_PATH.joinpath("time_abalation_act")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "time": list(),
        TRAIN: list(),
        TEST: list(),
    }
    train_loader, test_loader, _ = cifar10_loader(batch_size=512)
    for t in range(2, 101, 2):
        print(f"Running for epoch time {t}")
        vgg_model1, vgg_model2 = vgg16_bn(), vgg16_bn()
        vgg_model1.load_state_dict(
            torch.load(_STASH_PATH.joinpath(f"vgg16_bn1_{t}_100.pth"))
        )
        vgg_model1.to(DEVICE)
        vgg_model1.eval()

        vgg_model2.load_state_dict(
            torch.load(_STASH_PATH.joinpath(f"vgg16_bn2_{t}_100.pth"))
        )
        vgg_model2.to(DEVICE)
        vgg_model2.eval()

        permuter = ActMatching(arch=LOOK_UP_LAYER)
        model1_dict, model2_dict = dict(), dict()
        register_hook(inst=vgg_model1, activations_dict=model1_dict)
        register_hook(inst=vgg_model2, activations_dict=model2_dict)

        # TODO: Time the below two methods and get error value
        # Method 1: Evaluating cost matrix batch wise, values are
        # added element wise
        for inp, lbl in train_loader:
            _ = vgg_model1(inp.to(DEVICE))
            _ = vgg_model2(inp.to(DEVICE))

            # The dictionaries gets erased and updated every time
            permuter.evaluate_permutation(model1_dict, model2_dict)

        # Fetching the permutation
        _permutation_dict = permuter.get_permutation()

        # weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
        # _permutation_dict = weight_matcher.evaluate_permutation(
        #     m1_weights=mlp_model1.state_dict(), m2_weights=mlp_model2.state_dict()
        # )

        write_file(abalation_path.joinpath(f"{str(t)}.pkl"), _permutation_dict)

        vgg_model1, vgg_model2 = vgg16_bn(), vgg16_bn()
        vgg_model1.load_state_dict(
            torch.load(_STASH_PATH.joinpath(f"vgg16_bn1_{t}_100.pth"))
        )
        vgg_model1.to(DEVICE)
        vgg_model1.eval()

        vgg_model2.load_state_dict(
            torch.load(_STASH_PATH.joinpath(f"vgg16_bn2_{t}_100.pth"))
        )
        vgg_model2.to(DEVICE)
        vgg_model2.eval()

        results_dict = generate_plots(
            model1=vgg_model1,
            model2=vgg_model2,
            act_perm=_permutation_dict,
        )
        # plt_dict(results=results_dict, file_path=parent_folder.joinpath(f"vgg_epoch_act_{str(t)}"))

        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(
                    LAMBDA_ARRAY,
                    loss_arr,
                    label=method + "_" + set,
                    **fmt[set],
                    **fmt[method],
                )

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Act match: Epoch {t}")
        plt.savefig(parent_folder.joinpath(f"vgg_epoch_act_{str(t)}"))
        plt.clf()

        loss_barrier["time"].append(t)
        loss_barrier[TRAIN].append(
            max(results_dict[ACT_MATCH][TRAIN])
            - 0.5
            * (results_dict[ACT_MATCH][TRAIN][0] + results_dict[ACT_MATCH][TRAIN][-1])
        )

        loss_barrier[TEST].append(
            max(results_dict[ACT_MATCH][TEST])
            - 0.5
            * (results_dict[ACT_MATCH][TEST][0] + results_dict[ACT_MATCH][TEST][-1])
        )

    write_file(Path("loss_barrier_vgg_act_epoch.pkl"), loss_barrier)
    plt.figure()
    _fmt = {
        TRAIN: {"linestyle": "solid", "marker": "*"},
        TEST: {"linestyle": "dashed", "marker": "*"},
    }
    plt.plot(loss_barrier["time"], loss_barrier[TRAIN], label="Train", **_fmt[TRAIN])
    plt.plot(loss_barrier["time"], loss_barrier[TEST], label="Test", **_fmt[TEST])

    plt.xlabel("Epoch time")
    plt.ylabel("Loss Barrier")
    plt.legend()
    plt.savefig(f"Time_LossBarrier_vgg_act")
