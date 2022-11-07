import numpy
from scipy.optimize import linear_sum_assignment

from models.mlp_model import MLP


def loss_barrier(combined_model:MLP, model1:MLP, model2:MLP, loss_fn,lambda_list:list[float]):
    # TODO: Write loss barrier @Adhithyan8
    pass


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


def combine_models(model1: MLP, model2: MLP, perm_dict: dict):
    """
    Combine models

    :param model1: _description_
    :type model1: MLP
    :param model2: _description_
    :type model2: MLP
    :param perm_dict: _description_
    :type perm_dict: dict
    :return: _description_
    :rtype: MLP
    """
    model3 = MLP()
    model1_state_dict = model1.state_dict()
    model2_state_dict = model2.state_dict()
    model3_state_dict = model3.state_dict()

    for key in model3_state_dict.keys():
        model3_state_dict[key] = (
            model1_state_dict[key]
            + perm_dict[key.split(".")[0]] @ model2_state_dict[key]
        )

    model3.load_state_dict(model3_state_dict)

    return model3
