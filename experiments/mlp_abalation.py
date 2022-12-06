from pathlib import Path
from config import _STASH_PATH, ACT_MATCH, DEVICE, LAMBDA_ARRAY, MLP_PERM_PATH, NAIVE_MATCH, TEST, TRAIN, WEIGHT_MATCH
import torch
from helper import gif_maker, plt_dict, write_file
from models.utils import cifar10_loader
from permuter._algo import WeightMatching, ActMatching

from models.mlp import LAYER_NAMES, MLP, WEIGHT_PERM_LOOKUP, register_hook
from permuter.mlp import generate_plots
from matplotlib import pyplot as plt

fmt = {
            TRAIN: {"linestyle": "solid", "marker": "*"},
            TEST: {"linestyle": "dashed", "marker": "*"},
            NAIVE_MATCH: {"color": "k"},
            ACT_MATCH: {"color": "r"},
            WEIGHT_MATCH: {"color": "g"},
        }

def mlp_width_ablation():
    """
    weight matching

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    parent_folder = Path("mlp_width_plot")
    parent_folder.mkdir(exist_ok=True,parents=True)
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
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_{w}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_{w}_40.pth")))
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
        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(LAMBDA_ARRAY, loss_arr, label=method + "_" + set, **fmt[set], **fmt[method])           

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Weight match: Width {w}")
        plt.savefig(parent_folder.joinpath(f"mlp_width_{w:04}"))
        plt.clf()
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
    gif_maker(parent_folder)


def mlp_width_act_ablation():
    """
    weight matching

    :return: Permutation dictionary
    :rtype: dict[str, torch.Tensor]
    """
    parent_folder = Path("mlp_width_act_plot")
    parent_folder.mkdir(exist_ok=True,parents=True)
    abalation_path = MLP_PERM_PATH.joinpath("width_act_abalation")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "width":list(),
        TRAIN:list(),
        TEST:list(),
    }
    train_loader, test_loader, _ = cifar10_loader(batch_size=512)
    for w in [64,128,256,512,1024]:
        print(f"Running for width {w}")
        mlp_model1, mlp_model2 = MLP(WIDTH=w), MLP(WIDTH=w)
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_{w}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_{w}_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()

        permuter = ActMatching(arch=LAYER_NAMES)
        model1_dict, model2_dict = dict(), dict()
        register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
        register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

        # TODO: Time the below two methods and get error value
        # Method 1: Evaluating cost matrix batch wise, values are
        # added element wise
        for inp, lbl in train_loader:
            _ = mlp_model1(inp.to(DEVICE))
            _ = mlp_model2(inp.to(DEVICE))

            # The dictionaries gets erased and updated every time
            permuter.evaluate_permutation(model1_dict, model2_dict)

        # Fetching the permutation
        _permutation_dict = permuter.get_permutation()
        
        write_file(abalation_path.joinpath(f"{str(w)}.pkl"),_permutation_dict)
        
        mlp_model1, mlp_model2 = MLP(WIDTH=w), MLP(WIDTH=w)
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_{w}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_{w}_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()
        
        results_dict = generate_plots(
            model1=mlp_model1,
            model2=mlp_model2,
            weight_perm=_permutation_dict,
            width=w
        )
        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(LAMBDA_ARRAY, loss_arr, label=method + "_" + set, **fmt[set], **fmt[method])           

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Weight match: Width {w}")
        plt.savefig(parent_folder.joinpath(f"mlp_width_act_{w:04}"))
        plt.clf()
        loss_barrier["width"].append(w)
        loss_barrier[TRAIN].append(
            max(results_dict[WEIGHT_MATCH][TRAIN]) - 0.5*(results_dict[WEIGHT_MATCH][TRAIN][0]+results_dict[WEIGHT_MATCH][TRAIN][-1])
        )
           
        loss_barrier[TEST].append(
            max(results_dict[WEIGHT_MATCH][TEST]) - 0.5*(results_dict[WEIGHT_MATCH][TEST][0]+results_dict[WEIGHT_MATCH][TEST][-1])
        )
        
    write_file(Path("loss_barrier_mlp_width_act.pkl"), loss_barrier)
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
    plt.savefig(f"Width_act_LossBarrier_mlp")
    gif_maker(parent_folder)


def mlp_time_ablation():
    parent_folder = Path("mlp_epoch_plot")
    parent_folder.mkdir(exist_ok=True,parents=True)
    abalation_path = MLP_PERM_PATH.joinpath("time_abalation")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "time":list(),
        TRAIN:list(),
        TEST:list(),
    }
    
    for t in range(1,41):
        print(f"Running for epoch time {t}")
        mlp_model1, mlp_model2 = MLP(), MLP()
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_512_{t}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_512_{t}_40.pth")))
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
        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(LAMBDA_ARRAY, loss_arr, label=method + "_" + set, **fmt[set], **fmt[method])           

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Weight match: Epoch {t}")
        plt.savefig(parent_folder.joinpath(f"mlp_epoch_{t:02}"))
        plt.clf()
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
    gif_maker(parent_folder)
    

def mlp_act_time_ablation():
    parent_folder = Path("mlp_epoch_act_plot")
    parent_folder.mkdir(exist_ok=True,parents=True)
    abalation_path = MLP_PERM_PATH.joinpath("time_abalation_act")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "time":list(),
        TRAIN:list(),
        TEST:list(),
    }
    train_loader, test_loader, _ = cifar10_loader(batch_size=512)
    for t in range(1,41):
        print(f"Running for epoch time {t}")
        mlp_model1, mlp_model2 = MLP(), MLP()
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_512_{t}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_512_{t}_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()
        
        permuter = ActMatching(arch=LAYER_NAMES)
        model1_dict, model2_dict = dict(), dict()
        register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
        register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

        # TODO: Time the below two methods and get error value
        # Method 1: Evaluating cost matrix batch wise, values are
        # added element wise
        for inp, lbl in train_loader:
            _ = mlp_model1(inp.to(DEVICE))
            _ = mlp_model2(inp.to(DEVICE))

            # The dictionaries gets erased and updated every time
            permuter.evaluate_permutation(model1_dict, model2_dict)

        # Fetching the permutation
        _permutation_dict = permuter.get_permutation()

        # weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
        # _permutation_dict = weight_matcher.evaluate_permutation(
        #     m1_weights=mlp_model1.state_dict(), m2_weights=mlp_model2.state_dict()
        # )
        
        write_file(abalation_path.joinpath(f"{str(t)}.pkl"),_permutation_dict)
        
        mlp_model1, mlp_model2 = MLP(), MLP()
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_512_{t}_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_512_{t}_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()
        
        results_dict = generate_plots(
            model1=mlp_model1,
            model2=mlp_model2,
            act_perm=_permutation_dict,
        )
        # plt_dict(results=results_dict, file_path=parent_folder.joinpath(f"mlp_epoch_act_{str(t)}"))
        
        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(LAMBDA_ARRAY, loss_arr, label=method + "_" + set, **fmt[set], **fmt[method])           

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Act match: Epoch {t}")
        plt.savefig(parent_folder.joinpath(f"mlp_epoch_{t:02}"))
        plt.clf()
        
        loss_barrier["time"].append(t)
        loss_barrier[TRAIN].append(
            max(results_dict[ACT_MATCH][TRAIN]) - 0.5*(results_dict[ACT_MATCH][TRAIN][0]+results_dict[ACT_MATCH][TRAIN][-1])
        )
           
        loss_barrier[TEST].append(
            max(results_dict[ACT_MATCH][TEST]) - 0.5*(results_dict[ACT_MATCH][TEST][0]+results_dict[ACT_MATCH][TEST][-1])
        )
        
    write_file(Path("loss_barrier_mlp_act_epoch.pkl"), loss_barrier)
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
    plt.savefig(f"Time_LossBarrier_mlp_act")
    gif_maker(parent_folder)


def mlp_act_batch_ablation():
    parent_folder = Path("mlp_batch_act_plot")
    parent_folder.mkdir(exist_ok=True,parents=True)
    abalation_path = MLP_PERM_PATH.joinpath("time_abalation_act_batch")
    abalation_path.mkdir(exist_ok=True, parents=True)
    loss_barrier = {
        "time":list(),
        TRAIN:list(),
        TEST:list(),
    }
    train_loader, test_loader, _ = cifar10_loader(batch_size=512)
    for t in range(31):
        print(f"Running for epoch time {t}")
        mlp_model1, mlp_model2 = MLP(), MLP()
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_512_{t}_1_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_512_{t}_1_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()
        
        permuter = ActMatching(arch=LAYER_NAMES)
        model1_dict, model2_dict = dict(), dict()
        register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
        register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)

        # TODO: Time the below two methods and get error value
        # Method 1: Evaluating cost matrix batch wise, values are
        # added element wise
        for inp, lbl in train_loader:
            _ = mlp_model1(inp.to(DEVICE))
            _ = mlp_model2(inp.to(DEVICE))

            # The dictionaries gets erased and updated every time
            permuter.evaluate_permutation(model1_dict, model2_dict)

        # Fetching the permutation
        _permutation_dict = permuter.get_permutation()

        # weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
        # _permutation_dict = weight_matcher.evaluate_permutation(
        #     m1_weights=mlp_model1.state_dict(), m2_weights=mlp_model2.state_dict()
        # )
        
        write_file(abalation_path.joinpath(f"{str(t)}.pkl"),_permutation_dict)
        
        mlp_model1, mlp_model2 = MLP(), MLP()
        mlp_model1.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp1_512_{t}_1_40.pth")))
        mlp_model1.to(DEVICE)
        mlp_model1.eval()

        mlp_model2.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp2_512_{t}_1_40.pth")))
        mlp_model2.to(DEVICE)
        mlp_model2.eval()
        
        results_dict = generate_plots(
            model1=mlp_model1,
            model2=mlp_model2,
            act_perm=_permutation_dict,
        )
        
        plt.figure()
        for method, res in results_dict.items():
            for set, loss_arr in res.items():
                plt.plot(LAMBDA_ARRAY, loss_arr, label=method + "_" + set, **fmt[set], **fmt[method])           

        plt.xlabel("Lambda")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Act match: Batch {t}")
        plt.savefig(parent_folder.joinpath(f"mlp_act_batch_{t:02}"))
        plt.clf()
        
        loss_barrier["time"].append(t)
        loss_barrier[TRAIN].append(
            max(results_dict[ACT_MATCH][TRAIN]) - 0.5*(results_dict[ACT_MATCH][TRAIN][0]+results_dict[ACT_MATCH][TRAIN][-1])
        )
           
        loss_barrier[TEST].append(
            max(results_dict[ACT_MATCH][TEST]) - 0.5*(results_dict[ACT_MATCH][TEST][0]+results_dict[ACT_MATCH][TEST][-1])
        )
        
    write_file(Path("loss_barrier_mlp_act_batch.pkl"), loss_barrier)
    plt.figure()
    _fmt = {
        TRAIN: {"linestyle": "solid", "marker": "*"},
        TEST: {"linestyle": "dashed", "marker": "*"}
        }
    plt.plot(loss_barrier["time"], loss_barrier[TRAIN], label="Train", **_fmt[TRAIN])
    plt.plot(loss_barrier["time"], loss_barrier[TEST], label="Test", **_fmt[TEST])

    plt.xlabel("Batch #")
    plt.ylabel("Loss Barrier")
    plt.legend()
    plt.savefig(f"Time_LossBarrier_mlp_act_batch")
    gif_maker(parent_folder)    