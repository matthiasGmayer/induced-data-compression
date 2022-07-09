#!/bin/python3
from induced_data_compression.data import get_data_loaders
from induced_data_compression.model import train, Net
import os
import torch
from glob import glob
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_models():
    data_loaders = get_data_loaders()
    num_models = 10_000

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    hidden_layers = 3
    width = 5
    s = f"h{hidden_layers}w{width}"

    mkdir("../models")
    mkdir(f"../models/{s}")
    mkdir(f"../models/{s}/simple")
    mkdir(f"../models/{s}/complex")


    # for func_type in ["simple", "complex"]:
    for func_type in ["complex"]:
        offset = max((int(os.path.basename(s)[:-3]) for s in glob(f"../models/{s}/{func_type}/*")),default=-1)+1
        for i in range(offset, num_models):
            unsuccessful = True
            while unsuccessful:
                net = Net(width=width, hidden_layers=hidden_layers).to(device)
                trainloader = data_loaders[func_type, "train"]
                success,_,_ = train(net, trainloader)
                unsuccessful = not success
            torch.save(net, f"../models/{s}/{func_type}/{i}.pt")

if __name__ == '__main__':
    train_models()

