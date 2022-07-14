#!/bin/python3
import itertools
import random

import numpy as np
import torch
from scipy.stats import gaussian_kde

from evaluation import get_activations_from_loaded, get_mutual_information_for_activations, all_mutual_total_and_ratio_information
from matplotlib import pyplot as plt

from induced_data_compression.helper import load_info
from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn

device = "cuda" if torch.cuda.is_available() else "cpu"

def density_scatter( x , y, ax = None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    plt.scatter( x, y, c=z, **kwargs )


    return ax
def plot_density(x,y):
    xy = np.vstack([y])
    z = gaussian_kde(xy)(xy)
    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]
    plt.scatter(x, y, c=z, s=50)

def plot_scatter(lines,title):
    # for p in lines:
    #     plt.scatter(range(4),p)
    y=np.array(lines)
    for i in range(4):
        # plot_density([i]*len(lines),y[:,i])
        x= np.array([i] * len(lines))
        density_scatter(x, y[:, i])
    plt.gca().set_ylim(ymin=0)
    plt.title(title)
    plt.savefig(f"plots/{title}.png")
    plt.show()

def plot_lines(lines,title):
    for p in lines:
        plt.plot(p)

    # plt.gca().set_ylim(ymin=0)
    plt.title(title)
    plt.savefig(f"plots/{title}.png")
    plt.show()

def quantiles(layer, n=5):
    m = len(layer)
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
    elif mode == "scatter":
        plot_scatter(lines,title)


def plot_same_layer_different_networks(mode="quantiles",info_type="mutual"):
    load_range=30
    for func_type in ["simple", "complex"]:
        mut_info, tot_info, ratio_info = load_info(func_type, 30)

        if info_type == "ratio":
            info=ratio_info
        elif info_type == "total":
            info=tot_info
        elif info_type=="mutual":
            info=mut_info

        activation_layers = hidden_layers+1
        to_plot = []
        for act1,act2 in itertools.combinations(range(load_range),2):
            infs = info[act1,act2,range(activation_layers), range(activation_layers)]
            to_plot.append(infs.tolist())

        comment = "mi_same_layer_different_network"
        title = f"{func_type}_{load_range}_tol{tol}_{mode}_{info_type}_{comment}"
        plot(to_plot, title, mode)


def plot_information_each_layer(mode="quantiles", info_type="mutual"):
    for func_type in ["simple", "complex"]:
        mut_info, tot_info, ratio_info = load_info(func_type, 30)
        if info_type == "ratio":
            info = ratio_info
        elif info_type == "total":
            info = tot_info
        elif info_type == "mutual":
            info = mut_info
        # activations = get_activations_from_loaded(type, hidden_layers=hidden_layers,
        #                                           width=width, num_bits=num_bits, load_range=load_range)
        # get_mutual_information_for_activations(activations, tol=1e-5)
        activation_layers = hidden_layers+1
        to_plot = []
        for act1,act2 in zip(range(30),range(30)):
            infs = info[act1,act2,range(activation_layers), range(activation_layers)]
            to_plot.append(infs.tolist())

        comment = "mi_information_each_layer"
        title = f"{func_type}_{30}_tol{tol}_{mode}_{info_type}_{comment}"
        plot(to_plot, title, mode)

def plot_similarity():
    for func_type in ["simple", "complex"]:
        mut_info, tot_info, ratio_info = load_info(func_type, 30)
        # similarity = torch.mean(ratio_info, dim=[2, 3])
        similarity = torch.sum(torch.amax(ratio_info, dim=2),dim=2)+ torch.sum(torch.amax(ratio_info, dim=3),dim=2)
        # plot_lines([sorted(l.tolist()) for l in similarity],title=f"{func_type}_similarity_30_sorted")
        # plot_lines([l.tolist() for l in similarity],title=f"{func_type}_similarity_30_unsorted")

        title = f"{func_type}_similarity_30_histogram"
        plt.title(title)
        plt.xlim([3,8])
        # for i in range(1):
        #     sim = similarity[i].tolist()
        #     plt.hist(sim, bins=16)
        plt.hist(similarity.flatten().tolist(),bins=16)
        plt.savefig(f"{title}.png")
        plt.show()


def plot_3d(func,title, n=10):
    step = 1/(n-1)
    x,y = torch.meshgrid(torch.arange(0,1+step/2,step), torch.arange(0,1+step/2, step), indexing="xy")
    xf = x.flatten()
    yf = y.flatten()
    xd = xf.to(device)
    yd = yf.to(device)
    inputs = torch.cat((xd[None],yd[None])).T
    zf = func(inputs).to('cpu')
    z = zf.unflatten(dim=0, sizes=(n,n))

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(x.numpy(),y.numpy(),z.numpy())
    ax.set_title(title)
    plt.savefig(f"{title}.png")
    plt.show()







if __name__ == '__main__':
    num_bits = 5
    hidden_layers = 3
    width = 5
    load_range = 100
    tol=1e-5

    func = lambda x:x[:,0]*x[:,1]
    plot_3d(func,"x*y")

    # mut_info, tot_info, ratio_info = load_info("simple", 30)
    #
    # r0 = ratio_info[0,0]
    # r1 =ratio_info[0,26]
    # print(torch.mean(r0))
    # print(torch.mean(r1))
    # print(r0)
    # print(r1)


    # for func_type in ["complex","simple"]:
    #     mut_info, tot_info, ratio_info = load_info(func_type, 30)
    #     similiarity = torch.mean(ratio_info, dim=[2, 3])
    #     print(similiarity[range(30),range(30)].detach().numpy())



    # plot_similarity()
    # for info_type in ["total","mutual","ratio"]:
        # plot_same_layer_different_networks("scatter",info_type=info_type)
        # plot_information_each_layer("scatter",info_type=info_type)
