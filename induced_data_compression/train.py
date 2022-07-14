#!/bin/python3
from induced_data_compression.data import get_data_loaders
from induced_data_compression.model import train, Net
import os
import torch
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
_dataset_path = "../datasets"


def train_models(dataset="projection_and_random", load_saved=False, num_models=10000, tol=1e-7, hidden_layers=3,
                 width=5, replace=False, input_num_bits=5, cut_off=20000, lr=0.01,momentum=0.9, func_types=["simple","complex"]):
    data_loaders = get_data_loaders(save_path=f"{_dataset_path}", dataset=dataset, func_types=func_types)

    s = f"h{hidden_layers}w{width}"
    save_path = f"../models/{dataset}"

    if func_types is None:
        os.makedirs(f"{save_path}/{s}", exist_ok=True)
        func_types = [""]
        train_loaders = {"":data_loaders["train"]}
    else:
        for func_type in func_types:
            os.makedirs(f"{save_path}/{s}/{func_type}", exist_ok=True)
        func_types = ["/"+f for f in func_types]
        train_loaders = {f:data_loaders[f, "train"] for f in func_types}

    for func_type in func_types:
        if not load_saved and not replace:
            offset = max((int(os.path.basename(s)[:-3]) for s in glob(f"{save_path}/{s}/{func_type}/*")),
                         default=-1) + 1
        else:
            offset = 0
        for i in range(offset, num_models):
            model_path = f"{save_path}/{s}{func_type}/{i}.pt"
            max_attempts = 10
            for k in range(max_attempts):
                if load_saved and os.path.exists(model_path):
                    net = torch.load(model_path)
                else:
                    net = Net(width=width, hidden_layers=hidden_layers, input_num_bits=input_num_bits).to(device)
                success, _, _ = train(net, train_loaders[func_type], abs_tol=tol, cut_off=cut_off, lr=lr, momentum=momentum)
                if success:
                    break
            if k == max_attempts - 1 and not success:
                print(f"{model_path} was not trained successfully")
            else:
                torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    train_models(num_models=3, dataset="xor",func_types=None, hidden_layers=1, width=2, input_num_bits=2, cut_off=100000, lr=0.01, momentum=0.5)
    """
    h1w5:
    gets stuck at .0013, 1e-5
    h1w10:
    get stuck at 1e-5 or 1e-6
    """
