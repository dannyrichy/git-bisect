import copy
from functools import reduce
from config import _STASH_PATH, DEVICE, LAMBDA_ARRAY, TEST, TRAIN
from helper import create_calibration_curve, normalised_cost
from models.mlp import LAYER_NAMES, MLP, WEIGHT_PERM_LOOKUP
import torch
from models.utils import cifar10_loader
from permuter._algo import WeightMatching
from permuter.common import combine_models, get_losses
from permuter.mlp import permute_model
from matplotlib import pyplot as plt

class Ensemble(torch.nn.Module):
    def __init__(self, *models) -> None:
        super().__init__()
        self.models = models
    
    def forward(self, x):
        out_logits = torch.mean(torch.stack([model(x) for model in self.models]), dim=0)
        return out_logits



def combine_many_models(*models:MLP):
    """
    Combine multiple models

    :param model1: Model 1
    :type model1: torch.nn.Module
    :param model2: Model 2
    :type model2: torch.nn.Module
    :param lam: Lambda value in linear interpolation way
    :type lam: float
    :return: Combined model
    :rtype: torch.nn.Module
    """
    # Creating dummy model
    combined_model = copy.deepcopy(models[0]).to(DEVICE)
    combined_model_sd = combined_model.state_dict()
    list_sd = [model.state_dict() for model in models ]
    

    for key in combined_model_sd.keys():
        combined_model_sd[key] = (1/len(models)) * reduce(lambda x,y: torch.add(x, y), [sd[key] for sd in list_sd])

    combined_model.load_state_dict(combined_model_sd)
    return combined_model

def get_permuted_model(model1, model2):
    weight_matcher = WeightMatching(arch=LAYER_NAMES, perm_lookup=WEIGHT_PERM_LOOKUP)
    perm_dict = weight_matcher.evaluate_permutation(
            m1_weights=model1.state_dict(), m2_weights=model2.state_dict()
        )
    permuted_model = permute_model(model=model2,  perm_dict=perm_dict)
    return permuted_model

if __name__ == "__main__":
    models = list()
    NUM_MODELS = 5
    for i in range(NUM_MODELS):
        m = MLP().to(DEVICE)
        m.load_state_dict(torch.load(_STASH_PATH.joinpath(f"mlp{i+1}_512_40.pth")))
        models.append(m)
    
    # Getting permutation_dict for each model
    _permutation_model = [
        get_permuted_model(models[0], models[j])
        for j in range(1,NUM_MODELS)
        ]
    
    
    train_loader, test_loader, _ = cifar10_loader(batch_size=128)
    ensemble_model = Ensemble(*models)
    for ix, model in enumerate(models):
        x, y = create_calibration_curve(model=model, dataloader=train_loader)
        plt.plot(x, y, color='r',linestyle="--",marker="*", label="Individual models" if ix==0 else "")
    x, y = create_calibration_curve(model=ensemble_model, dataloader=train_loader)
    plt.plot(x, y, color='b', marker="*", label="Ensemble model")
    def _generate_models(model1: torch.nn.Module, model2: torch.nn.Module) -> tuple:
        """
        Internal function to ensure temporary tensors gets erased

        :param _model2: Model 2
        :type _model2: torch.nn.Module
        :return: Result dictionary
        :rtype: dict[str, np.ndarray]
        """
        _models = list()
        for lam in LAMBDA_ARRAY:
            tmp = combine_models(model1=model1, model2=model2, lam=lam)
            tmp.eval()
            _models.append(tmp)
        _res = {
            TRAIN: get_losses(
                data_loader=train_loader,
                combined_models=_models,
            ),
            TEST: get_losses(
                data_loader=test_loader,
                combined_models=_models,
            ),
        }
        # pprint.pprint(_res)
        return max(_res[TRAIN]) - 0.5*(_res[TRAIN][0] + _res[TRAIN][-1]), max(_res[TEST]) - 0.5*(_res[TEST][0] + _res[TEST][-1])
    
    print("model1 and permuted_model2",_generate_models(models[0], _permutation_model[0]), normalised_cost(w1=models[0].state_dict(), w2= _permutation_model[0].state_dict()))
    print("model1 and permuted_model3",_generate_models(models[0], _permutation_model[1]), normalised_cost(w1=models[0].state_dict(), w2= _permutation_model[1].state_dict()))
    print("model1 and permuted_model4",_generate_models(models[0], _permutation_model[2]), normalised_cost(w1=models[0].state_dict(), w2= _permutation_model[2].state_dict()))
    print("model1 and permuted_model5",_generate_models(models[0], _permutation_model[3]), normalised_cost(w1=models[0].state_dict(), w2= _permutation_model[3].state_dict()))
    
    # for i in range(4):
    #     for j in range(i+1,4):
    #         print(f"{i+1}&{j+1}",_generate_models(_permutation_model[i], _permutation_model[j]))
    # print("Done !")
    
    # print("Permuting 3*,4*,5* to 2*")
    # _perm_second_order  = [
    #     get_permuted_model(_permutation_model[0], models[j])
    #     for j in range(1,NUM_MODELS-1)
    #     ]
    
    # print("model 2* & model 3**",_generate_models(_permutation_model[0], _perm_second_order[0]))
    # print("model 2* & model 4**",_generate_models(_permutation_model[0], _perm_second_order[1]))
    # print("model 2* & model 5**",_generate_models(_permutation_model[0], _perm_second_order[2]))
    # for i in range(3):
    #     print(f"model 1 & model {i+3}**",_generate_models(models[0], _perm_second_order[i]))
    #     for j in range(i+1,3):
    #         print(f"model {i+3}** & model {j+3}**",_generate_models(_perm_second_order[i], _perm_second_order[j]))
            
    print("Permuting all to centroid")
    __mix_models = [models[0]] + _permutation_model 
    for _ in range(10):
        for i in range(5):
            _model_tmp = combine_many_models(*__mix_models[:i], *__mix_models[i+1:])
            __mix_models[i] = get_permuted_model(_model_tmp,  __mix_models[i])
    
    merged_model_final =combine_many_models(*__mix_models)
    x, y = create_calibration_curve(model=merged_model_final, dataloader=train_loader)
    
    plt.plot(x, y, color='g',marker="*", label="Merged model")          
    plt.plot([0, 1], [0, 1], color='black', label="Perfect calibration")
    plt.xlabel('Predicted probability')
    plt.ylabel('Actual probabiliyt')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig("calibration_plot")
    
    merged_model_corr = 0.0
    individual_model_corr = [0.0 for _ in range(models.__len__())]
    ensemble_model_corr = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            
            # Merged model
            _, preds = torch.max(merged_model_final(inputs.to(DEVICE)), dim=1)
            merged_model_corr += torch.sum(preds == labels.to(DEVICE)).item()
            
            # Individual models
            for ix, m in enumerate(models):
                _, preds = torch.max(m(inputs.to(DEVICE)), dim=1)
                individual_model_corr[ix] += torch.sum(preds == labels.to(DEVICE)).item() 
            
            # Merged model
            _, preds = torch.max(ensemble_model(inputs.to(DEVICE)), dim=1)
            ensemble_model_corr += torch.sum(preds == labels.to(DEVICE)).item()              
        
        print("Merged model accuracy: ", merged_model_corr / len(test_loader.dataset))  #type: ignore
        print("Individual model accuracies: ", [tmp / len(test_loader.dataset) for tmp in individual_model_corr])  #type: ignore
        print("Ensemble model accuracy: ", ensemble_model_corr / len(test_loader.dataset))  #type: ignore
            
    
    print("Done!")