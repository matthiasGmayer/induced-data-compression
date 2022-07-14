#!/bin/python3
import numpy as np
import torch
import math
import random
import os
import torch.utils.data as data_utils
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import DataLoader
from data import get_data_loaders


device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, input_num_bits=5, hidden_layers=1, width=5):
        super(Net, self).__init__()
        self.sequential = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_num_bits, width),
                nn.ReLU()
            ),
            *(nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU()
            ) for _ in range(hidden_layers)),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.sequential(x)

class Net_with_activations(nn.Module):
    def __init__(self, input_num_bits=5, hidden_layers=1, width=5):
        super(Net_with_activations, self).__init__()
        self.sequential = nn.Sequential(
            nn.Sequential(
                nn.Linear(input_num_bits, width),
                nn.ReLU()
            ),
            *(nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU()
            ) for _ in range(hidden_layers)),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return [(x := l(x)) for l in self.sequential]


def load_models(func_type, width, hidden_layers, load_range, input_num_bits, save_path="../models", dataset="projection_and_random"):
    """
    load all models in "{save_path}/h{hidden_layers}w{width}/{func_type}/{r}" where r is in load_range
    If one model is not found, the function returns.
    :param func_type: e.g. "complex", "simple"
    :param width: the width of the net
    :param hidden_layers: the num. of hidden layers of the net
    :param load_range: Either int to load range(load_range) or a sequence to load all models in the sequence
    :param save_path: the path to the folder where the models are saved
    :return: the loaded models in a list
    """
    if isinstance(load_range, int):
        load_range = range(load_range)
    path = f"{save_path}/{dataset}/h{hidden_layers}w{width}/{func_type}"
    models = []
    for i in load_range:
        try:
            net=Net(width=width,hidden_layers=hidden_layers,input_num_bits=input_num_bits)
            net.load_state_dict(torch.load(f"{path}/{i}.pt"))
            models.append(net)
        except FileNotFoundError as e:
            print(e)
            print(f"Attempt at loading {i}th model failed, returning {i} loaded models")
            break
    return models

def train(net, train_loader, abs_tol=1e-05, cut_off=20_000 ,lr=0.01, momentum=0.9):
    """
  trains a net to zero loss
  Returns (success, epoch, loss):
  success: True iff zero loss was reached in  < cut_off episodes
  epoch: the epoch in which zero loss was reached
  loss: the actual loss reached
  Parameters:
  abs_tol: the tolerance for when we consider the loss = 0
  cut_off: the cut off when we consider the network stuck
  """
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=momentum)
    criterion = nn.MSELoss()
    loss = 1
    epoch = -1
    unsuccessful = False
    while not math.isclose(loss, 0, abs_tol=abs_tol):
        epoch += 1
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.to(device))[:, 0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 500 == 0:
            print(f"epoch={epoch}, loss={loss}")
        if epoch > cut_off:
            unsuccessful = True
            break
    return not unsuccessful, epoch, loss


def optimize_input_for_activations(net, input,
                                   locations,
                                   # get_activations_to_maximize,
                                   epochs=20000, lr=0.01, momentum=0.9):
    optimizer = torch.optim.SGD([input], lr=lr, momentum=momentum)
    net.requires_grad_(False)
    loss = 1
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = net(input.to(device)[None])
        s = torch.tensor(0.).to(device)
        for a in locations:
            s += torch.sum(output[a[0]][:, a[1:]])
        loss = -s
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"epoch={epoch}, -loss={-loss}")
            # print(input)
            # print(output)
    return -loss


def print_output(output):
    l = [o.to("cpu").tolist() for o in output]
    for i in l:
        for a in i:
            print(f"{a:00.2f}", end=" ")
        print()
if __name__ == '__main__':
    state_dict = torch.load("../models/fuzzy/h3w10/simple/0.pt")
    net = Net_with_activations(input_num_bits=3,hidden_layers=3,width=10)
    net.load_state_dict(state_dict)
    net.to(device)
    state_dict2 = torch.load("../models/fuzzy/h3w10/simple/1.pt")
    net2 = Net_with_activations(input_num_bits=3,hidden_layers=3,width=10)
    net2.load_state_dict(state_dict2)
    net2.to(device)
    input = nn.Parameter(torch.tensor([ 1.,1.,1. ]).to(device))
    locations = [(0,0)]
    optimize_input_for_activations(net,input,locations, epochs=10000)
    print(input)
    print_output(net(input))
    print_output(net2(input))
