#!/bin/python3
import itertools
import random

import torch

from evaluation import get_activations_from_loaded, get_mutual_information_for_activations, all_mutual_total_and_ratio_information
from matplotlib import pyplot as plt

from induced_data_compression.helper import load_info


def plot_lines(lines,title):
    for p in lines:
        plt.plot(p)

    plt.gca().set_ylim(ymin=0)
    plt.title(title)
    plt.savefig(f"{title}.png")
    plt.show()

def quantiles(layer, n=5):
    print("hello")
    m = len(layer)
    print(layer)
    sl = sorted(layer)
    return [sl[(m//n)*i] for i in range(1,n)]


def plot_quantiles(lines, title):

    layers = list(zip(*lines))
    quants = [quantiles(layer, n=5) for layer in layers]
    lines = list(zip(*quants))
    plot_lines(lines, title)

def plot(lines, title, mode):
    if mode == "quantiles":
        plot_quantiles(lines, title)
    elif mode == "lines":
        plot_lines(lines,title)


def plot_same_layer_different_networks(mode="quantiles"):
    for type in ["simple", "complex"]:
        activations = get_activations_from_loaded(type, hidden_layers=hidden_layers,
                                                  width=width, num_bits=num_bits, load_range=load_range)
        activation_layers = hidden_layers+1
        to_plot = []
        for act1,act2 in itertools.combinations(activations,2):
            mut_info, tot_info, ratio_info = all_mutual_total_and_ratio_information(act1,act2,tol=tol)
            infs= mut_info[range(activation_layers), range(activation_layers)]
            to_plot.append(infs.tolist())

        comment = "mi_same_layer_different_network"
        title = f"{type}_{load_range}_tol{tol}_{mode}_{comment}"
        plot(to_plot, title, mode)


def plot_information_each_layer(mode="quantiles"):
    for type in ["simple", "complex"]:
        activations = get_activations_from_loaded(type, hidden_layers=hidden_layers,
                                                  width=width, num_bits=num_bits, load_range=load_range)
        get_mutual_information_for_activations(activations, tol=1e-5)
        activation_layers = hidden_layers+1
        to_plot = []
        for act in activations:
            mut_info, tot_info, ratio_info = all_mutual_total_and_ratio_information(act,act,tol=tol)
            infs = mut_info[range(activation_layers), range(activation_layers)]
            to_plot.append(infs.tolist())

        comment = "mi_information_each_layer"
        title = f"{type}_{load_range}_tol{tol}_{mode}_{comment}"
        plot(to_plot, title, mode)

def plot_similarity():
    for func_type in ["simple", "complex"]:
        mut_info, tot_info, ratio_info = load_info(func_type, 30)
        # similarity = torch.mean(ratio_info, dim=[2, 3])
        similarity = torch.sum(torch.amax(ratio_info, dim=2),dim=2)+ torch.sum(torch.amax(ratio_info, dim=3),dim=2)
        plot_lines([sorted(l.tolist()) for l in similarity],title=f"{func_type}_similarity_30_sorted")
        plot_lines([l.tolist() for l in similarity],title=f"{func_type}_similarity_30_unsorted")
        plt.
        plt.hist(similarity)




if __name__ == '__main__':
    num_bits = 5
    hidden_layers = 3
    width = 5
    load_range = 100
    tol=1e-5

    mut_info, tot_info, ratio_info = load_info("simple", 30)

    r0 = ratio_info[0,0]
    r1 =ratio_info[0,26]
    print(torch.mean(r0))
    print(torch.mean(r1))
    print(r0)
    print(r1)


    # for func_type in ["complex","simple"]:
    #     mut_info, tot_info, ratio_info = load_info(func_type, 30)
    #     similiarity = torch.mean(ratio_info, dim=[2, 3])
    #     print(similiarity[range(30),range(30)].detach().numpy())



    plot_similarity()
    # plot_same_layer_different_networks()
    # plot_information_each_layer()

