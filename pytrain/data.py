# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import torch


class Batch:
    def __init__(self, data=None, target=None):
        self.data = data
        self.target = target

    def to(self, device):
        if self.data is not None:
            self.data = self.data.to(device)
        if self.target is not None:
            self.target = self.target.to(device)
        return self


class Dataset:
    def __init__(self, training, validation=None, testing=None):
        self.training = training
        self.validation = validation
        self.testing = testing

    @classmethod
    def from_data(_, data, train_split=0.9):
        if hasattr(data, "__next__") or hasattr(data, "__getitem__"):
            return Dataset(data)

        if isinstance(data, Dataset):
            return data

        if isinstance(data, tuple):
            if len(data) != 3:
                raise ValueError("Dataset requires tuple of length three.")
            return Dataset(*data)

        if isinstance(data, torch.Tensor):
            split = int(data.shape[0] * 0.9)
            return Dataset(data[:split], data[split:], None)

        raise TypeError(f"Unknown type `{type(data)}` for Dataset.")
