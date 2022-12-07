import pathlib
import pickle
import time
from typing import Any

import numpy
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt
import imageio.v3 as iio
import imageio

from config import (
    ACT_MATCH,
    DEVICE,
    LAMBDA_ARRAY,
    NAIVE_MATCH,
    STE_MATCH,
    TEST,
    TIME_FLAG,
    TRAIN,
    WEIGHT_MATCH,
)


def timer_func(name: str):
    """
    Decorator function to note down the time taken to execute

    :param name: Name of the function that is to be noted
    :type name: str
    """

    def inner_func(func):
        def wrap_func(*args, **kwargs):
            if TIME_FLAG:
                t1 = time.time()
                result = func(*args, **kwargs)
                print(f"Method {name} executed in {(time.time()-t1):.4f}s")
            else:
                result = func(*args, **kwargs)
            return result

        return wrap_func

    return inner_func


def write_file(file_path: pathlib.Path, obj: Any) -> None:
    """
    Write everything as a pickle file

    :param file_path: _description_
    :type file_path: str
    :param obj: _description_
    :type obj: Any
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_file(file_path: pathlib.Path) -> Any:
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def plt_dict(results: dict[str, dict[str, numpy.ndarray]],file_path:pathlib.Path) -> None:
    plt.figure()
    _fmt = {
        TRAIN: {"linestyle": "solid", "marker": "*"},
        TEST: {"linestyle": "dashed", "marker": "*"},
        NAIVE_MATCH: {"color": "k"},
        ACT_MATCH: {"color": "r"},
        WEIGHT_MATCH: {"color": "g"},
        STE_MATCH: {"color": "b"},
    }
    
    for method, res in results.items():
        for set, loss_arr in res.items():
            plt.plot(LAMBDA_ARRAY, loss_arr, label=method + "_" + set, **_fmt[set], **_fmt[method])
            

    plt.xlabel("Lambda")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_path)


def gif_maker(folder_path:pathlib.Path):
    images = list()
    for file in sorted(folder_path.iterdir()):
        if not file.is_file():
            continue

        images.append(iio.imread(file))
    
    imageio.mimsave(str(folder_path) + ".gif", images, fps=2)