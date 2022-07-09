#!/bin/python3
import torch
import math
import random
import os
import torch.utils.data as data_utils
from collections import defaultdict
import torch.nn as nn
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"


class Net(nn.Module):
    def __init__(self, input_num_bits=5, hidden_layers=1, width=5):
        super(Net, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_num_bits, width),
            nn.ReLU(),
            *(nn.Sequential(
                nn.Linear(width, width),
                nn.ReLU()
            ) for _ in range(hidden_layers)),
            nn.Linear(width, 1)
        )

    def forward(self, x):
        return self.sequential(x)


def train(net, train_loader, abs_tol=1e-05, cut_off=20_000):
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
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
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
