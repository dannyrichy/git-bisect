import pathlib
import pickle
import time
from typing import Any

import numpy
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader 
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt

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


def plt_dict(results: dict[str, dict[str, numpy.ndarray]]) -> None:
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
    plt.savefig("Results_" + time.strftime("%Y%m%d-%H%M%S"))


# Create pytorch code for calibration curve give dataloader and model
def create_calibration_curve(model, dataloader, file_path,num_bins=10):
    # Get binned predictions
    bin_boundaries = numpy.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    with torch.no_grad():
        _bins = []
        _truth = []
        for data, label in dataloader:
            output = torch.nn.functional.softmax(model(data.to(DEVICE)), dim=1)
            _tmp = numpy.digitize(output.cpu().numpy(), bin_boundaries) - 1
            _max = numpy.max(_tmp, axis=1)
            _true_val = numpy.argmax(_tmp, axis=1) == label.numpy()
            _bins.append(_max)
            _truth.append(_true_val)
        _bins = numpy.concatenate(_bins)
        _truth = numpy.concatenate(_truth)
        _agg_truth = [numpy.sum(_truth[_bins == i])/_truth[_bins == i].shape[0] for i in range(num_bins)]
        plt.plot(bin_lowers, _agg_truth, color='g')          
        plt.plot([0, 1], [0, 1], color='black', label="Perfect")
        plt.xlabel('Predicted probability')
        plt.ylabel('Actual probabiliyt')
        plt.title('Calibration Curve')
        plt.savefig(file_path)
        
