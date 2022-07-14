#!/bin/python3
from induced_data_compression.data import get_data_loaders
from induced_data_compression.model import train, Net
import os
import torch
from glob import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
_dataset_path = "../datasets"


def train_models(dataset="projection_and_random", load_saved=False, num_models=10000, tol=1e-7, hidden_layers=3,
                 width=5, replace=False, input_num_bits=5, cut_off=20000, lr=0.01,momentum=0.9):
    data_loaders = get_data_loaders(save_path=f"{_dataset_path}", dataset=dataset)

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    s = f"h{hidden_layers}w{width}"
    save_path = f"../models/{dataset}"

    mkdir(f"{save_path}")
    mkdir(f"{save_path}/{s}")
    mkdir(f"{save_path}/{s}/simple")
    mkdir(f"{save_path}/{s}/complex")

    # for func_type in ["simple", "complex"]:
    for func_type in ["simple"]:
        if not load_saved and not replace:
            offset = max((int(os.path.basename(s)[:-3]) for s in glob(f"{save_path}/{s}/{func_type}/*")),
                         default=-1) + 1
        else:
            offset = 0
        for i in range(offset, num_models):
            model_path = f"{save_path}/{s}/{func_type}/{i}.pt"
            max_attempts = 10
            for k in range(max_attempts):
                if load_saved and os.path.exists(model_path):
                    net = torch.load(model_path)
                else:
                    net = Net(width=width, hidden_layers=hidden_layers, input_num_bits=input_num_bits).to(device)
                train_loader = data_loaders[func_type, "train"]
                success, _, _ = train(net, train_loader, abs_tol=tol, cut_off=cut_off, lr=lr, momentum=momentum)
                if success:
                    break
            if k == max_attempts - 1 and not success:
                print(f"{model_path} was not trained successfully")
            else:
                torch.save(net.state_dict(), model_path)


if __name__ == '__main__':
    train_models(num_models=3, dataset="fuzzy", hidden_layers=3, width=10, input_num_bits=3, cut_off=50000, lr=0.01, momentum=0.9)
    """
    h1w5:
    gets stuck at .0013, 1e-5
    h1w10:
    get stuck at 1e-5 or 1e-6
    """
