import torch
import os
import random
from induced_data_compression.data import get_inputs

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

def fuzzy_not(a: float) -> float:
    return 1 - a

def fuzzy_or(a: float, b: float) -> float:
    return max(a, b)

def fuzzy_and(a: float, b: float) -> float:
    return min(a, b)

def fuzzy_xor(a: float, b: float) -> float:
    return fuzzy_and(fuzzy_or(a, b), fuzzy_not(fuzzy_and(a, b)))

def fuzzy_implies(a: float, b: float) -> float:
    return fuzzy_or(fuzzy_not(a), b)

def get_continuous_inputs(num_samples: int, num_inputs: int = 3) -> torch.Tensor:
    """For now, uniformly randomly sample in [0,1] for each input and add the 8 boolean examples"""
    inputs = torch.rand(num_samples, num_inputs)
    print(inputs.shape, get_inputs(num_bits=num_inputs).shape)
    inputs = torch.cat((inputs, get_inputs(num_bits=num_inputs)))

    return inputs

def simple_fuzzy_func(inputs: torch.Tensor) -> torch.Tensor:
    return torch.tensor([fuzzy_implies(a.item(), fuzzy_implies(b.item(), c.item())) for (a, b, c) in inputs])

def complex_fuzzy_func(inputs: torch.Tensor) -> torch.Tensor:
    return torch.tensor([fuzzy_xor(a.item(), fuzzy_xor(b.item(), c.item())) for (a, b, c) in inputs])

def generate_datasets(train_size: int = 4, test_size: int = 4, num_bits: int = 3, save_path="../datasets"):
    """
    generates and saves the datasets in {save_path}
    """
    # note that we subtract the number of bits we'll have to use
    n_samples = train_size + test_size - 2**num_bits
    if n_samples < 0:
        raise ValueError("train_size + test_size must be greater than 2**num_bits")
    inputs = get_continuous_inputs(train_size + test_size - 2**num_bits, num_inputs=num_bits)
    indices = list(range(len(inputs)))
    random.shuffle(indices)
    inputs = inputs[indices]
    train_inputs, test_inputs = inputs[test_size:], inputs[:test_size]
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(train_inputs, f"{save_path}/fuzzy_train_inputs.pt")
    torch.save(test_inputs, f"{save_path}/fuzzy_test_inputs.pt")
    print(inputs)
    print(simple_fuzzy_func(train_inputs))
    print(complex_fuzzy_func(train_inputs))
    torch.save(simple_fuzzy_func(train_inputs), f"{save_path}/train_fuzzy_simple_truths.pt")
    torch.save(simple_fuzzy_func(test_inputs), f"{save_path}/test_fuzzy_simple_truths.pt")
    torch.save(complex_fuzzy_func(train_inputs), f"{save_path}/train_fuzzy_complex_truths.pt")
    torch.save(complex_fuzzy_func(test_inputs), f"{save_path}/test_fuzzy_complex_truths.pt")


def main():
    # inputs = get_inputs(num_bits=3)
    generate_datasets(train_size=12, test_size=4, num_bits=3)
    # simple_outputs = t.tensor([fuzzy_implies(a, fuzzy_implies(b, c)) for (a, b, c) in inputs])
    # complex_outputs = t.tensor([fuzzy_xor(a, fuzzy_xor(b, c)) for (a, b, c) in inputs])
    # print(simple_outputs)
    # print(complex_outputs)


if __name__ == "__main__":
    main()

