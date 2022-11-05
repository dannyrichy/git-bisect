import numpy
import torch
from procrustes import permutation

from helper import timer_func


class Permuter:
    """"
    Parent class for all the methods
    """
    perm = list()
    loss = list()
    def __init__(self, archi:list[int], model_width):
        self.archi = archi
        self.model_width = model_width


class ActivationMethod(Permuter):
    def __init__(self, archi, model_width) -> None:
        super().__init__(archi, model_width)

    def _layer_wise(self, act_a:numpy.ndarray, act_b:numpy.ndarray)->None:
        """
        Model B is considered to be permuted

        :param act_a: Activation of Model a layer in the model
        :type act_a: numpy.ndarray
        :param act_b: Activation of Model b layer in the model
        :type act_b: numpy.ndarray
        """
        # Note that the procrustes algorithm works with the form
        
        res = permutation(act_b.T, act_a)
        self.loss.append(res.get('error'))
        self.perm.append(res.get('t').T) # type: ignore
    
    @timer_func("Activation method")
    def get_permuation(self, model_a_act:dict[str, torch.Tensor], model_b_act:dict[str, torch.Tensor]):
        """
        Get's layer wise permutation matrix
        
        :param model_a_act: Activations of all layers of model A
        :type model_a_act: dict
        
        :param model_b_act: Activations of all layers of model B
        :type model_b_act: dict
        
        :return: List of permutation matrix
        :rtype: list[numpy.ndarray]
        """ 
        if len(self.loss) !=0 :
            self.loss = list()
            self.perm = list()
        for act_a, act_b in zip(model_a_act.values(), model_b_act.values()):
            self._layer_wise(act_a=act_a.detach().numpy(), act_b= act_b.detach().numpy())
        
        return self.perm

    def get_loss(self):
        """
        Gets loss value of the operation

        :return: Loss 
        :rtype: float
        """
        return self.loss

class GreedyAlgorithm(Permuter):
    def __init__(self, archi: list[int], model_width) -> None:
        """
        _summary_

        :param archi: _description_
        :type archi: list[int]
        :param model_width: _description_
        :type model_width: _type_
        """
        super().__init__(archi, model_width)
    
    def get_permutation(self, model_a, model_b):
        """
        Generate permutation for the weights of model B

        :param model_a: Model A
        :type model_a: torch.nn
        
        :param model_b: Model B
        :type model_b: torch.nn
        
        :return: List of permutation of each layer
        :rtype: 
        """
    
        
        return numpy.zeros((2,2))