import torch

from config import MLP_MODEL1_PATH, MLP_MODEL2_PATH
from models.mlp_model import MLP, register_hook


def fetch_activations(data_loader):
  mlp_model1, mlp_model2 = MLP(), MLP()
  mlp_model1.load_state_dict(torch.load(MLP_MODEL1_PATH))
  mlp_model1.eval()
  
  mlp_model2.load_state_dict(torch.load(MLP_MODEL2_PATH))
  mlp_model2.eval()
  
  model1_dict, model2_dict = dict(), dict()
  register_hook(mlp_inst=mlp_model1, activations_dict=model1_dict)
  register_hook(mlp_inst=mlp_model2, activations_dict=model2_dict)
  
  for inp, lbl in data_loader:
      _ = mlp_model1(inp)
      _ = mlp_model2(inp)
      break 
  
  return model1_dict, model2_dict