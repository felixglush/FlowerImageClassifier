import torch

def get_device(gpu):
    return torch.device("cuda:0" if (gpu and torch.cuda.is_available()) else "cpu")