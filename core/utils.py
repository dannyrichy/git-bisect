from copy import copy
from typing import Union

import numpy
import torch

from models.mlp_model import MLP


class LossBarrier:
    def __init__(
        self,
        model1: MLP,
        model2: MLP,
        lambda_list: Union[numpy.ndarray, list[float]],
        perm_dict: dict[str, numpy.ndarray],
    ) -> None:
        self.model1 = model1
        self.model2 = model2
        self.permuted_model2 = self._permute_model2(model2)
        self.lambda_list = lambda_list
        self.perm_dict = perm_dict

    def _permute_model2(self, model2: MLP) -> MLP:
        permuted_model = MLP()
        perm_state_dict = permuted_model.state_dict()
        model2_state_dict = model2.state_dict()
        for key in perm_state_dict.keys():
            layer_name, weight_type = key.split(".")
            if weight_type == "weight" and not layer_name.startswith("1"):
                _layer_name, _layer_num = layer_name.split("_")
                prev_layer_name = "_".join([_layer_name, str(int(_layer_num) - 1)])
                perm_state_dict[key] = (
                    torch.Tensor(self.perm_dict[layer_name])
                    @ model2_state_dict[key]
                    @ torch.Tensor(self.perm_dict[prev_layer_name]).T
                )
            else:
                perm_state_dict[key] = (
                    torch.Tensor(self.perm_dict[layer_name]) @ model2_state_dict[key]
                )
        permuted_model.load_state_dict(perm_state_dict)
        permuted_model.eval()
        return permuted_model

    def _common_combination(self, state_dict1, state_dict2, lam):
        model3 = MLP()
        model3_state_dict = model3.state_dict()

        for key in model3_state_dict.keys():
            model3_state_dict[key] = (1 - lam) * state_dict1[key] + lam * state_dict2[
                key
            ]

        model3.load_state_dict(model3_state_dict)
        model3.eval()
        return model3

    def naive_combine_models(self, lam: float = 0.5):
        """
        Combine models without permutation

        :param lam: lambda value to combine models by, defaults to 0.5
        :type lam: float, optional
        :return: Combined models
        :rtype: list[dict]
        """
        return self._common_combination(
            self.model1.state_dict(), self.model2.state_dict(), lam
        )

    def combine_models(self, lam: float = 0.5) -> MLP:
        """
        Combines models model1 and model2 as (1-lam)*model1 + lam*model2

        :param lam: lambda value to combine models by, defaults to 0.5
        :type lam: float, optional
        :return: combined model
        :rtype: MLP
        """
        return self._common_combination(
            self.model1.state_dict(), self.permuted_model2.state_dict(), lam
        )

    def loss_barrier(self, data_loader: torch.util.data.DataLoader):  # type: ignore
        """
        Returns function to calculate loss barrier for all values in lambda_list. Ensure that the batch_size is high enough

        :param data_loader:
        :type data_loader:
        :return: _description_
        :rtype: _type_
        """

        loss_barrier_dict = dict()
        for inp, out in data_loader:
            _sum_losses = 0.5 * (
                torch.nn.functional.cross_entropy(self.model1(inp), out)
                + torch.nn.functional.cross_entropy(self.model2(inp), out)
            )
            loss_barrier_dict["Activation matching"] = loss_barrier_dict.get(
                "Activation matching", 0
            ) + numpy.array(
                [
                    (
                        torch.nn.functional.cross_entropy(
                            self.combine_models(_lam_val)(inp), out
                        )
                        - _sum_losses
                    ).item()
                    for _lam_val in self.lambda_list
                ]
            )
            loss_barrier_dict["Naive matching"] = loss_barrier_dict.get(
                "Naive matching", 0
            ) + numpy.array(
                [
                    (
                        torch.nn.functional.cross_entropy(
                            self.naive_combine_models(_lam_val)(inp), out
                        )
                        - _sum_losses
                    ).item()
                    for _lam_val in self.lambda_list
                ]
            )

        return loss_barrier_dict
