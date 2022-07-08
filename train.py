#!/bin/python3
from data import get_data_loaders
from model import train, Net
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

    mkdir("models")
    mkdir("models/simple")
    mkdir("models/complex")

    offset = max((int(os.path.basename(s)[:-2]) for s in glob("models/simple/*")),default=-1)+1

    for func_type in ["simple", "complex"]:
        for i in range(offset, num_models):
            unsuccessful = True
            while unsuccessful:
                net = Net().to(device)
                trainloader = data_loaders[func_type, "train"]
                success,_,_ = train(net, trainloader)
                unsuccessful = not success
            torch.save(net, f"models/{func_type}/{i}.pt")

if __name__ == '__main__':
    train_models()

