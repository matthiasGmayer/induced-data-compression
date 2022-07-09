#!/bin/python3
import itertools
import math
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
    return torch.max(torch.abs(tensor_a - tensor_b)).item() < tol


def buckets(indices, func, tol):
    # idea: sort the activations into buckets according to 'nearness', i.e. consider near activations the same
    """
    sorts the indices into buckets s.t. for all a,b in a bucket there exists a=c_1 .. c_n=b unique in the bucket with
    dis(c_i,c_(i+1)) < tol
    :param indices: the indices which to bucket
    :param func: maps indices to tensors
    :param tol: the tolerance for when two tensors are considered equal.
    :return: a dict: indices -> bucket, where the bucket is a list. If two indices are in the same bucket they point
    towards the same list object.
    """
    indices_to_bucket = {index: {index} for index in indices}

    def union(i, j):
        if indices_to_bucket[i] is not indices_to_bucket[j]:
            indices_to_bucket[i] |= indices_to_bucket[j]
            indices_to_bucket[j] = indices_to_bucket[i]

    for i in indices:
        for j in indices:
            if i != j and is_near(func(i), func(j), tol=tol):
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


def mutual_information(buckets_a, buckets_b):
    """
    computes mutual_information given the buckets of the marginal and joint distributions
    :param buckets_a:
    :param buckets_b:
    :param buckets_ab:
    :return: the mutual information
    """
    # unique buckets correspond to a discrete value in the image of a
    a_unique_buckets = list(unique(buckets_a.values()))
    b_unique_buckets = list(unique(buckets_b.values()))
    buckets_ab = joint_bucket(a_unique_buckets, b_unique_buckets)
    total_elements = len(buckets_a)
    sum = 0
    for a in a_unique_buckets:
        for b in b_unique_buckets:
            p_ab = len(buckets_ab[id(a), id(b)]) / total_elements
            if p_ab == 0:
                continue
            p_a = len(a) / total_elements
            p_b = len(b) / total_elements
            val = p_ab * math.log2(p_ab / p_a / p_b)
            sum += val
    return sum


def all_mutual_information(activations_a, activations_b, tol=1e-3):
    """
    Assumes that the inputs are discretely uniformly distributed,
    calculates the mutual information of all (x,y) \in X \times Y,
    where X, Y are the set of all layer activations
    :param activations_a: The activations of X of shape (num_inputs, num_activation_layers, width)
    :param activations_b: The activation of Y of shape
    :return the mutual information for each pair of layers
    """
    num_inputs, num_activations, width = activations_a.shape
    mut_info = torch.zeros((num_activations, num_activations))
    index_range = range(num_inputs)
    all_buckets_a = []
    for layer_a in range(num_activations):
        activation_a = activations_a[:, layer_a, :]
        all_buckets_a.append(buckets(index_range, lambda index: activation_a[index, :], tol=tol))
    all_buckets_b = []
    for layer_b in range(num_activations):
        activation_b = activations_b[:, layer_b, :]
        all_buckets_b.append(buckets(index_range, lambda index: activation_b[index, :], tol=tol))
    for layer_a in range(num_activations):
        for layer_b in range(num_activations):
            mut_info[layer_a, layer_b] = mutual_information(all_buckets_a[layer_a], all_buckets_b[layer_b])
    return mut_info

def get_activations_from_loaded(model_type,hidden_layers,width,load_range,num_bits):
    """
    loads all the models of the specified type in the specified range.
    :param load_range: either int or sequence
    :return: the activations for each model as a list
    """
    num_inputs=2**num_bits
    activation_layers = hidden_layers+1
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


def get_mutual_information_for_activations(activations):
    """
    :param activations: the activations of the models
    :return: tensor of shape (m,m,a,a) where m=num_models, a=num_activation_layers
    e.g. mut_info[1,2] is the mutal information between all the layers of the first and second model
    """
    num_models = len(activations)
    num_activation_layers = activations[0].shape[1]
    mutual_information_list = torch.zeros((num_models, num_models, num_activation_layers, num_activation_layers))
    for i in range(num_models):
        for j in range(i+1,num_models):
            mut_info = all_mutual_information(activations[i], activations[j])
            mutual_information_list[i, j] = mut_info
            mutual_information_list[j, i] = mut_info.T
    return mutual_information_list


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

    num_bits = 5
    hidden_layers = 3
    width = 5
    load_range = 2
    activations = get_activations_from_loaded("simple", hidden_layers=hidden_layers,
                                              width=width,num_bits=num_bits, load_range=load_range)
    all_mut_info_list = get_mutual_information_for_activations(activations)
    print(all_mut_info_list)
