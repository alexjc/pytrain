# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import torch


class BasicTrainer:
    def __init__(self, parameters, lr=1e-2):
        self.optimizer = torch.optim.Adam(parameters, lr=lr)

    def step(self, task, args):
        self.optimizer.zero_grad()
        loss = task.function(*args)
        loss.backward()
        self.optimizer.step()
        return loss.item()
