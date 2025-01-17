# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import torch


class Batch:
    """Data-class that stores multiple items sampled from a dataset.
    """

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.attributes = list(kwargs.keys())
        self.number = None

    @classmethod
    def from_data(_, data):
        if isinstance(data, Batch):
            return data
        if isinstance(data, dict):
            return Batch(**data)
        return Batch(data=data)

    def to(self, device):
        for key in self.attributes:
            try:
                setattr(self, key, getattr(self, key).to(device))
            except AttributeError:
                pass
        return self


class Dataset:
    """Container for a dataset that's split into training, validation (optional)
    and testing (optional) segments.
    """

    def __init__(self, training, validation=None, testing=None):
        self.training = training
        self.validation = validation
        self.testing = testing

    @classmethod
    def from_data(_, data, train_split=0.9):
        if isinstance(data, Dataset):
            return data

        if isinstance(data, (tuple, list)):
            if len(data) != 3:
                raise ValueError("Dataset requires tuple of length three.")
            return Dataset(*data)

        if hasattr(data, "__next__") or hasattr(data, "__getitem__"):
            return Dataset(data)

        if isinstance(data, torch.Tensor):
            split = int(data.shape[0] * train_split)
            return Dataset(data[:split], data[split:], None)

        raise TypeError(f"Unknown type `{type(data)}` for Dataset.")
