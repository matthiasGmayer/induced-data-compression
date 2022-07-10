#!/bin/python3
import random

from evaluation import get_activations_from_loaded, get_mutual_information_for_activations, all_mutual_total_and_ratio_information
from matplotlib import pyplot as plt

if __name__ == '__main__':
    num_bits = 5
    hidden_layers = 3
    width = 5
    load_range = 100
    type = "complex"
    activations = get_activations_from_loaded(type, hidden_layers=hidden_layers,
                                              width=width, num_bits=num_bits, load_range=load_range)
    activation_layers = hidden_layers+1
    to_plot = []
    for act1,act2 in zip(activations[:-1], activations[1:]):
        mut_info, tot_info, ratio_info = all_mutual_total_and_ratio_information(act1,act2)
        infs= mut_info[range(activation_layers), range(activation_layers)]
        to_plot.append(infs.tolist())

    comment = "line"
    for p in to_plot:
        plt.plot(p)
    plt.savefig(f"{type}_{load_range}_{comment}.png")
    plt.show()

