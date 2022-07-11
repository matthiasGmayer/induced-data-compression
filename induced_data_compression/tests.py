#!/bin/python3
from evaluation import get_activations_from_loaded, get_mutual_information_for_activations

if __name__ == '__main__':
    pass
    # TODO check correctness of total information
    # H(X,Y) = H(X)+H(Y)-MI(X,Y)

    # activations = get_activations_from_loaded("simple", 3, 5, 10, 5)
    # hidden_layers = 3
    # num_inputs = 2 ** 5
    # tol = 1e-5
    # act_buckets = []
    # for act in range(len(activations)):
    #     act_buckets.append([buckets(range(num_inputs), lambda i: activations[act][i, layer], tol=tol)
    #                         for layer in range(hidden_layers + 1)])
    # mi = []
    # for activations_a, activations_b in itertools.combinations(range(len(activations)), 2):
    #     l =[]
    #     for layer in range(hidden_layers + 1):
    #         m = mutual_information(act_buckets[activations_a][layer], act_buckets[activations_b][layer],
    #                            num_inputs=num_inputs)
    #         l.append(m)
    #     m.append(l)
    # print(m)

    tol = 1e-5
    load_range=3
    activations = get_activations_from_loaded("simple",3,5,load_range,5)
    mut_info, tot_info, ratio_info = get_mutual_information_for_activations(activations,tol=tol)

    r = range(load_range)
    r2 = range(4)
    for a in r:
        for b in r:
            for la in r2:
                for lb in r2:
                    d = tot_info[a,b,la,lb] - (mut_info[a,a,la,la]+mut_info[b,b,lb,lb]-mut_info[a,b,la,lb])
                    if d != 0:
                        print(d,a,b,la,lb)
    for a in r:
        for la in r2:
            d = tot_info[a,a,la,la] - mut_info[a,a,la,la]
            if d != 0:
                print(d, a, la)

