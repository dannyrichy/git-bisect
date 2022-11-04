from torch import nn


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self) -> None:
    super().__init__()
    self.layer_1 = nn.Linear(32 * 32 * 3, 64)
    self.relu_layer_1 = nn.ReLU()
    self.layer_2 = nn.Linear(64, 32)
    self.relu_layer_2 = nn.ReLU()
    self.layer_3 = nn.Linear(32, 10)
    


  def forward(self, x):
    """
    _summary_

    :param x: Input variable
    :type x: torch.Tensor
    
    :return: predicted output
    :rtype: torch.Tensor
    """
    y = self.layer_1(x)
    y = self.relu_layer_1(y)
    y = self.layer_2(y)
    y = self.relu_layer_2(y)
    y = self.layer_3(y)
        
    return y
