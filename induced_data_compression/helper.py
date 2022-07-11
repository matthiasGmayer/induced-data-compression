import torch


def unique(sequence, key=id):
    """
    :param sequence: the sequence to be filtered
    :param key: a function of the elements of the sequence. If the key is the same the elements are regarded as the same
    :return: yields the unique elements of the sequence (as a generator)
    """
    d = set()
    for s in sequence:
        if key(s) not in d:
            d.add(key(s))
            yield s

def load_info(type,load_range):
    return torch.load(f"info_{type}_{load_range}.pt")