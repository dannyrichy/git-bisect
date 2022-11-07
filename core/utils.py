import torch
import numpy
from scipy.optimize import linear_sum_assignment


def loss_barrier(model_a, model_b, loss_func, x, y, lambda_list=None):
    """
    Gets the list of loss barrier for provided list of lambda

    :param model_a:
    :type model_a: torch.Model

    :param model_b:
    :type model_b:

    :param loss_func:
    :type loss_func:

    :param x:
    :type x:

    :param y:
    :type y:

    :param lambda_list:
    :type lambda_list:

    :return:
    :rtype:
    """
    if lambda_list is None:
        lambda_list = [0.1 * i for i in range(1, 9)]

    return [
            loss_func(lam * model_a.predict(x) + (1 - lam) * model_b.predict(x), y) - 0.5 * (loss_func(model_a.predict(x), y) + loss_func(
                    model_b.predict(x), y))

            for lam in lambda_list
    ]


def compute_permutation_hungarian(cost_matrix: numpy.ndarray) -> numpy.ndarray:
        """
        _summary_

        :param cost_matrix: _description_
        :type cost_matrix: numpy.ndarray
        :return: _description_
        :rtype: numpy.ndarray
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        # make the permutation matrix by setting the corresponding elements to 1
        perm = numpy.zeros(cost_matrix.shape)
        perm[(row_ind, col_ind)] = 1
        return perm
        