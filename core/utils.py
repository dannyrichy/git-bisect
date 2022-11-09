from typing import Union

import numpy
import torch

from config import DEVICE
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
        self.lambda_list = lambda_list
        self.perm_dict = perm_dict
        self.permuted_model2 = self._permute_model2(model2)

    def _permute_model2(self, model2: MLP) -> MLP:
        permuted_model = MLP().to(DEVICE)
        perm_state_dict = permuted_model.state_dict()
        model2_state_dict = model2.state_dict()
        for key in perm_state_dict.keys():
            layer_name, weight_type = key.split(".")
            if weight_type == "weight" and not layer_name.endswith("1"):
                _layer_name, _layer_num = layer_name.split("_")
                prev_layer_name = "_".join([_layer_name, str(int(_layer_num) - 1)])
                perm_state_dict[key] = (
                    torch.Tensor(self.perm_dict[layer_name]).to(DEVICE)
                    @ model2_state_dict[key]
                    @ torch.Tensor(self.perm_dict[prev_layer_name]).to(DEVICE).T
                )
            else:
                perm_state_dict[key] = (
                    torch.Tensor(self.perm_dict[layer_name]).to(DEVICE)
                    @ model2_state_dict[key]
                )
        permuted_model.load_state_dict(perm_state_dict)
        permuted_model.eval()
        return permuted_model

    def _common_combination(self, state_dict1, state_dict2, lam):
        model3 = MLP().to(DEVICE)
        model3_state_dict = model3.state_dict()

        for key in model3_state_dict.keys():
            model3_state_dict[key] = (1 - lam) * state_dict1[key] + lam * state_dict2[
                key
            ]

        model3.load_state_dict(model3_state_dict)
        model3.eval()
        return model3

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

    def loss_barrier(self, data_loader: torch.utils.data.DataLoader):  # type: ignore
        """
        Returns function to calculate loss barrier for all values in lambda_list. Ensure that the batch_size is high enough

        :param data_loader:
        :type data_loader:
        :return: _description_
        :rtype: _type_
        """
        naive_models = [
            self._common_combination(
                self.model1.state_dict(), self.model2.state_dict(), lam
            )
            for lam in self.lambda_list
        ]
        combined_models = [
            self._common_combination(
                self.model1.state_dict(), self.permuted_model2.state_dict(), lam
            )
            for lam in self.lambda_list
        ]
        loss_barrier_dict = dict()
        counter = 0.0
        n_loss, p_loss = [0.0 for _ in range(len(self.lambda_list))], [
            0.0 for _ in range(len(self.lambda_list))
        ]
        for inp, out in data_loader:
            _sum_losses = (
                0.5
                * (
                    torch.nn.functional.cross_entropy(self.model1(inp.to(DEVICE)), out.to(DEVICE))
                    + torch.nn.functional.cross_entropy(self.model2(inp.to(DEVICE)), out.to(DEVICE))
                ).item()
            )

            for ix, (nm, pm) in enumerate(zip(naive_models, combined_models)):
                n_loss[ix] += (
                    torch.nn.functional.cross_entropy(nm(inp.to(DEVICE)), out.to(DEVICE)).item() - _sum_losses
                )
                p_loss[ix] += (
                    torch.nn.functional.cross_entropy(pm(inp.to(DEVICE)), out.to(DEVICE)).item() - _sum_losses
                )

            counter += 1.0

        loss_barrier_dict["Activation matching"] = numpy.array(p_loss) / counter
        loss_barrier_dict["Naive matching"] = numpy.array(n_loss) / counter

        return loss_barrier_dict
