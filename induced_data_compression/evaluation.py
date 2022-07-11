#!/bin/python3
import functools
import itertools
import math
import numpy
import os
import torch
from torch import nn
from model import load_models
from data import get_inputs
from helper import unique

device = "cuda" if torch.cuda.is_available() else "cpu"


def register_activation_saving_hook(model, activations):
    """
    registers forward hooks that save the activations inplace in the tensor "activations"
    :param model: the model where to save the activations
    :param activations: a tensor of shape (num_inputs, num_activation_layers, width)
    """
    l = [module for module in model.modules() if type(module) == nn.ReLU]

    def get_hook(layer):
        def hook(module, input, output):
            activations[:, layer, :].copy_(output)

        return hook

    for i, relu in enumerate(l):
        relu.register_forward_hook(get_hook(i))


def is_near(tensor_a, tensor_b, tol):
    """
    :param tensor_a
    :param tensor_b
    :param tol: the tolerance
    :return:
    """
    # max is probably better because the joint distribution need to be consistent with the marginal distr.
    # but the join has twice as many values
    # return settings.is_near_function(tensor_a, tensor_b, tol)
    setting = "l_1_mean"
    # setting = "l_inf"
    if setting == "l_inf":
        return torch.max(torch.abs(tensor_a - tensor_b)).item() < tol
    elif setting == "l_1_mean":
        return torch.mean(torch.abs(tensor_a - tensor_b)).item() < tol
    elif setting == "l_2_mean":
        return torch.mean(torch.abs(tensor_a - tensor_b) ** 2).item() < tol


def buckets(indices, func, tol, init_buckets=None):
    # idea: sort the activations into buckets according to 'nearness', i.e. consider near activations the same
    """
    sorts the indices into buckets s.t. for all a,b in a bucket there exists a=c_1 .. c_n=b unique in the bucket with
    dis(c_i,c_(i+1)) < tol
    :param indices: the indices which to bucket
    :param func: maps indices to tensors
    :param tol: the tolerance for when two tensors are considered equal.
    :param init_buckets: which inputs should be considered the same, e.g. to convert a uniform distribution of 1..16
    into one {1..8},{9..16}
    :return: a dict: indices -> bucket, where the bucket is a list. If two indices are in the same bucket they point
    towards the same list object.
    """
    if init_buckets == None:
        indices_to_bucket = {index: {index} for index in indices}
    else:
        indices_to_bucket = {index: next(b for b in init_buckets if index in b) for index in indices}

    def union(i, j):
        if indices_to_bucket[i] is not indices_to_bucket[j]:
            indices_to_bucket[i] |= indices_to_bucket[j]
            for b in indices_to_bucket[j]:
                indices_to_bucket[b] = indices_to_bucket[i]

    for i in indices:
        for j in indices:
            if i == j:
                continue
            tensor_i = func(i)
            tensor_j = func(j)
            if is_near(tensor_i, tensor_j, tol=tol):
                union(i, j)

    return indices_to_bucket


def joint_bucket(buckets_a, buckets_b):
    """
    :param buckets_a: a sequence of buckets as set
    :param buckets_b: a sequence of buckets as set
    :return: a dict (id of set, id of set) -> set, the joint buckets, namely the intersections of buckets_a x buckets_b
    """
    # joint buckets:
    # buckets of a correspond to the values of a
    # we have to look at all possible (bucket_a, bucket_b) which are values in our joint distribution
    # the probability of (bucket_a, bucket_b) is, with our assumption of a uniform dsitr. over the inputs,
    # the count the number of inputs i where (a(i),b(i)) \in (bucket_a,bucket_b)
    return {(id(a), id(b)): a & b for a in buckets_a for b in buckets_b}

def information(buckets, num_inputs=None):
    """
    computes information given the buckets of a distribution
    :param buckets: buckets are a dict element -> set of elements
    :return: the information contained
    """
    if num_inputs is None:
        num_inputs = sum(len(b) for b in buckets)
    total_elements = num_inputs
    s = 0
    for b in buckets:
        p = len(b) / total_elements
        val = -p * math.log2(p)
        s += val
    return s

def mutual_information(buckets_a, buckets_b, buckets_ab, num_inputs=None):
    """
    computes mutual_information given the buckets of the marginal and joint distributions
    :param buckets_a: buckets are a dict element -> set of elements
    :param buckets_b:
    :param buckets_ab:
    :return: the mutual information
    """
    # total_elements = len(buckets_a)
    if num_inputs is None:
        num_inputs = sum(len(b) for b in buckets_a)
    total_elements = num_inputs
    s = 0
    for a in buckets_a:
        for b in buckets_b:
            p_ab = len(buckets_ab[id(a), id(b)]) / total_elements
            if p_ab == 0:
                continue
            p_a = len(a) / total_elements
            p_b = len(b) / total_elements
            val = p_ab * math.log2(p_ab / p_a / p_b)
            s += val
    return s

def all_mutual_total_and_ratio_information(activations_a, activations_b, tol=1e-3):
    """
    Assumes that the inputs are discretely uniformly distributed,
    calculates the mutual information of all (x,y) \in X \times Y,
    where X, Y are the set of all layer activations
    :param activations_a: The activations of X of shape (num_inputs, num_activation_layers, width)
    :param activations_b: The activation of Y of shape
    :return the information for each pair of layers, ratio = -1 means information was 0
    """
    num_inputs, num_activations, width = activations_a.shape
    mut_info = torch.zeros((num_activations, num_activations))
    tot_info = torch.zeros((num_activations, num_activations))
    ratio_info =torch.zeros((num_activations, num_activations))
    index_range = range(num_inputs)
    all_buckets_a = []
    for layer_a in range(num_activations):
        all_buckets_a.append(buckets(index_range, lambda index: activations_a[index, layer_a, :], tol=tol))
    all_buckets_b = []
    for layer_b in range(num_activations):
        all_buckets_b.append(buckets(index_range, lambda index: activations_b[index,layer_b, :], tol=tol))
    for layer_a in range(num_activations):
        for layer_b in range(num_activations):
            buckets_a, buckets_b = all_buckets_a[layer_a], all_buckets_b[layer_b]
            # unique buckets correspond to a discrete value in the image of a
            buckets_a_unique = list(unique(buckets_a.values()))
            buckets_b_unique = list(unique(buckets_b.values()))
            buckets_ab = joint_bucket(buckets_a_unique, buckets_b_unique)
            mutual_inf = mutual_information(buckets_a_unique, buckets_b_unique, buckets_ab, num_inputs)
            buckets_a_b = buckets(index_range,
                                  lambda index: torch.cat((activations_a[index, layer_a, :],
                                                           activations_b[index, layer_b, :])),
                                  tol=tol)
            buckets_a_b_unique = list(unique(buckets_a_b.values()))
            inf = information(buckets_a_b_unique, num_inputs)
            ratio = mutual_inf/inf if inf > 0 else -1
            mut_info[layer_a, layer_b] = mutual_inf
            tot_info[layer_a, layer_b] = inf
            ratio_info[layer_a, layer_b] = ratio
    return mut_info, tot_info, ratio_info


def get_activations_from_loaded(model_type, hidden_layers, width, load_range, num_bits):
    """
    loads all the models of the specified type in the specified range.
    :param load_range: either int or sequence
    :return: the activations for each model as a list
    """
    num_inputs = 2 ** num_bits
    activation_layers = hidden_layers + 1
    models = load_models(model_type, hidden_layers=hidden_layers, width=width, load_range=load_range)
    activations = [torch.zeros((num_inputs, activation_layers, width)) for _ in range(len(models))]

    for i, model in enumerate(models):
        model.eval()
        register_activation_saving_hook(model, activations[i])
    with torch.no_grad():
        inputs = get_inputs(num_bits).to(device).to(torch.float)
        for model in models:
            model(inputs)
    return activations

def get_mutual_information_for_activations(activations,tol=1e-3):
    """
    :param activations: the activations of the models
    :return: tensor of shape (m,m,a,a) where m=num_models, a=num_activation_layers
    e.g. mut_info[1,2] is the mutal information between all the layers of the first and second model
    """
    num_models = len(activations)
    num_activation_layers = activations[0].shape[1]
    mutual_information_list = torch.zeros((num_models, num_models, num_activation_layers, num_activation_layers))
    total_information_list = torch.zeros((num_models, num_models, num_activation_layers, num_activation_layers))
    ratio_information_list = torch.zeros((num_models, num_models, num_activation_layers, num_activation_layers))
    def assign(i,j,input):
        mut_info, tot_info, ratio_info = input
        mutual_information_list[i, j] = mut_info
        mutual_information_list[j, i] = mut_info.T
        total_information_list[i, j] = tot_info
        total_information_list[j, i] = tot_info.T
        ratio_information_list[i, j] = ratio_info
        ratio_information_list[j, i] = ratio_info.T

    for i in range(num_models):
        print(f"{i}/{num_models}")
        for j in range(i + 1, num_models):
            output = all_mutual_total_and_ratio_information(activations[i], activations[j],tol=tol)
            assign(i,j, output)
    for i in range(num_models):
        output = all_mutual_total_and_ratio_information(activations[i], activations[i],tol=tol)
        assign(i, i, output)
    return mutual_information_list, total_information_list, ratio_information_list

if __name__ == '__main__':
    # tensors = [
    #     torch.tensor([0,0,0]),
    #     torch.tensor([0,0,0]),
    #     torch.tensor([0,0,0]),
    #     torch.tensor([0,0,1]),
    #     torch.tensor([0,0,1]),
    #     torch.tensor([1,2,1]),
    #     torch.tensor([1,4,3])
    # ]
    # indices = range(len(tensors))
    # b = buckets(indices,lambda t:tensors[t], tol = 0.01)


    # m,t,r="a\n"*200,"b\n"*200,"c\n"*200
    # lines = [str(a).splitlines() for a in [m,t,r]]
    # for l in zip(*lines):
    #     print(*l, sep="  ")

    # assert False
    num_bits = 5
    hidden_layers = 3
    width = 5
    load_range = 30
    for func_type in ["complex","simple"]:
        activations = get_activations_from_loaded(func_type, hidden_layers=hidden_layers,
                                                  width=width, num_bits=num_bits, load_range=load_range)
        info = get_mutual_information_for_activations(activations)
        torch.save(info, f"info_{func_type}_{load_range}.pt")
        mut_info, tot_info, ratio_info = info


    # print(mut_info<=tot_info)
    # assert torch.all(mut_info<=tot_info+1e-3)

    # a,b = [0,0,1],[1,2,2]
    # m = mut_info[a,b].detach().numpy()
    # t = tot_info[a,b].detach().numpy()
    # r = ratio_info[a,b].detach().numpy()
    # lines = [str(a).splitlines() for a in [m,t,r]]
    # for l in zip(*lines):
    #     print(*l, sep=" \t")


    # torch.set_printoptions(threshold=100_000)
    # with open("output", "w+") as f:
    #     f.write(str(all_mut_info_list))
