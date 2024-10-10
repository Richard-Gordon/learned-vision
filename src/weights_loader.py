import torch
from torchvision import models
import os

def load_weights(dir_name='./data/weights') -> dict:
    """ Load available model weights in the specified directory """
    weights = {}
    for filename in os.listdir(dir_name):
        if filename.endswith('.pt'):
            model_name = filename.split('.')[0]
            module = torch.load(f'{dir_name}/{filename}', weights_only=True)
            weights[model_name] = module['weight']
    return weights


def save_weights(dir_name='./data/weights', kernel_size=(3,3), stride=(2,2)):
    """ Get and save first-layer weights of specified type from all Torchvision models """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    model_weights = get_weights(models.list_models(), kernel_size, stride)
    for model_name, model_weights in model_weights.items():
        filename = f'{dir_name}/{model_name}.pt'
        torch.save(model_weights, filename)


def get_weights(model_list: str, kernel_size=(3,3), stride=(2,2)) -> dict:
    """ Get the first-layer weights of specified type from all Torchvision models """
    weights = {}
    for model_name in model_list:
        model = models.get_model(model_name, weights='DEFAULT')
        model_weights = get_model_weights(model, kernel_size, stride)
        if model_weights is not None:
            weights[model_name] = model_weights
    return weights


def get_model_weights(model: str, kernel_size=(3,3), stride=(2,2)) -> dict | None:
    """ Get the first-layer weights of specified type from the named model """
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.kernel_size == kernel_size and module.stride == stride:
                return module.state_dict()
            break
    return None




