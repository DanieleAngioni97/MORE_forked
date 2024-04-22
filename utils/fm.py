import os
import pickle
import torch

def my_load(path, format='rb', is_torch=True):
    if is_torch:
        return torch.load(path, map_location='cpu')
    with open(path, format) as f:
        data = pickle.load(path)
    return data