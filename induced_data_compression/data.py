#!/bin/python
import torch
import random
import os
import torch.utils.data as data_utils
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
_default_save_path = "../datasets"
_default_dataset = "projection_and_random"
_default_func_types = ["simple", "complex"]


def get_inputs(num_bits: int = 5) -> torch.Tensor:
    return torch.tensor([[int(char) for char in f"{i:0>{num_bits}b}"] for i in range(2 ** num_bits)], dtype=torch.int32)


def simple_func(bits):
    return bits[:, 0]


def complex_func(bits):
    return torch.tensor(random.choices(population=[0, 1], k=bits.shape[0]), dtype=torch.int32)


def generate_datasets(test_size=8, save_path=_default_save_path, dataset=_default_dataset):
    """
  generates and saves the datasets in {save_path}
  """
    num_bits = 5
    inputs = get_inputs(num_bits=num_bits)
    indices = list(range(len(inputs)))
    random.shuffle(indices)
    inputs = inputs[indices]
    train_inputs, test_inputs = inputs[test_size:], inputs[:test_size]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(train_inputs, f"{save_path}/{dataset}/train_inputs.pt")
    torch.save(test_inputs, f"{save_path}/{dataset}/test_inputs.pt")
    torch.save(simple_func(train_inputs), f"{save_path}/{dataset}/train_simple_truths.pt")
    torch.save(simple_func(test_inputs), f"{save_path}/{dataset}/test_simple_truths.pt")
    torch.save(complex_func(train_inputs), f"{save_path}/{dataset}/train_complex_truths.pt")
    torch.save(complex_func(test_inputs), f"{save_path}/{dataset}/test_complex_truths.pt")
    print(train_inputs)
    print(simple_func(train_inputs))


def _delete_datasets(save_path=_default_save_path, dataset=_default_dataset):
    for data_type in ["simple", "complex"]:
        for train_type in ["train", "test"]:
            os.remove(f"{save_path}/{dataset}/{train_type}_inputs.pt")
            os.remove(f"{save_path}/{dataset}/{train_type}_{data_type}_truths.pt")


def get_datasets(save_path=_default_save_path, dataset=_default_dataset, test_size=8, func_types=_default_func_types):
    """
  loads the datasets as torch.Dataset into a dict
  If there are no datasets presents, they are generated.
  Example access:
  datasets["simple","train"]
  Parameters:
  test_size: the test_size used for generate_datasets if new datasets are generated
  """
    datasets = {}
    if dataset == _default_dataset and not os.path.exists(save_path+"/"+dataset + "/train_inputs.pt"):
        generate_datasets(test_size, save_path)
    if func_types is None:
        for train_type in ["train", "test"]:
            inputs = torch.load(f"{save_path}/{dataset}/{train_type}_inputs.pt").to(torch.float).to(device)
            truths = torch.load(f"{save_path}/{dataset}/{train_type}_truths.pt").to(torch.float).to(device)
            print(inputs.shape, truths.shape)
            loaded_dataset = data_utils.TensorDataset(inputs, truths)
            datasets[train_type] = loaded_dataset
    else:
        for data_type in func_types:
            for train_type in ["train", "test"]:
                inputs = torch.load(f"{save_path}/{dataset}/{train_type}_inputs.pt").to(torch.float).to(device)
                truths = torch.load(f"{save_path}/{dataset}/{train_type}_{data_type}_truths.pt").to(torch.float).to(device)
                loaded_dataset = data_utils.TensorDataset(inputs, truths)
                datasets[data_type, train_type] = loaded_dataset
    return datasets


def get_data_loaders(batch_size=24, shuffle=False, datasets=None, save_path=_default_save_path, dataset=_default_dataset, func_types=_default_func_types):
    if datasets is None:
        datasets = get_datasets(save_path,dataset,func_types=func_types)
    return {key: DataLoader(batch_size=batch_size, shuffle=shuffle, dataset=dataset)
            for key, dataset in datasets.items()}


if __name__ == '__main__':
    # testing
    d = get_datasets()
    l = list(d["simple", "train"])
    assert len(l) == 24
    assert len(set([tuple(r) for r in l])) == 24
    for a, b in l:
        assert a[0].item() == b.item()
    l = list(d["simple", "test"])
    assert len(l) == 8
    assert len(set([tuple(r) for r in l])) == 8
    for a, b in l:
        assert a[0].item() == b.item()
