# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import torch


class Batch:
    def __init__(self, data, target=None):
        self.data = data
        self.target = target

    def to(self, device):
        self.data = self.data.to(device)
        if self.target is not None:
            self.target = self.target.to(device)
        return self


class Dataset:
    def __init__(self, training, validation, testing):
        self.training = training
        self.validation = validation
        self.testing = testing

    def length(self):
        return self.training.shape[0]

    @classmethod
    def from_data(_, data, train_split=0.9):
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
