# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import hashlib
import itertools

import torch

from .data import Batch


def iterate_ordered(data, batch_size=32):
    for i in itertools.count():
        indices = torch.arange(i * batch_size, i + 1 * batch_size, size=(batch_size,))
        yield Batch(input=data[indices % data.shape[0]])


def iterate_random(data, batch_size=32):
    while True:
        indices = torch.randint(0, data.shape[0], size=(batch_size,))
        yield Batch(input=data[indices])


class BasicTrainer:
    def __init__(self, task, args, params, lr=1e-2):
        for key in ("batch", "iterator"):
            if key in args:
                args[key] = iterate_random(args[key])

        self.args = args
        self.task = task

        self.optimizer = torch.optim.Adam(params, lr=lr)

    def step(self):
        args = self.args.copy()
        if "batch" in args:
            args["batch"] = next(args["batch"])

        self.optimizer.zero_grad()
        loss = self.task.function(**args)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, args):
        log = []
        for instance in args.values():
            if not hasattr(instance, "parameters"):
                continue

            cls = instance.__class__
            data = (cls.__module__ + "." + cls.__qualname__).encode("utf-8")
            digest = hashlib.blake2b(data, digest_size=8).hexdigest()
            filename = f"{cls.__name__}-{digest}.pkl"
            torch.save(instance, f"models/{filename}")
            log.append(cls.__qualname__)

        print("💾  Saved model snapshot for: {}.".format(", ".join(log)))
