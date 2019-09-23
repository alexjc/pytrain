# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import hashlib
import itertools

import torch

from .data import Batch


def iterate_ordered(data, batch_size):
    for i in itertools.count():
        indices = torch.arange(i * batch_size, i + 1 * batch_size, size=(batch_size,))
        yield Batch(input=data[indices % data.shape[0]])


def iterate_random(data, batch_size):
    while True:
        indices = torch.randint(0, data.shape[0], size=(batch_size,))
        yield Batch(input=data[indices])


class BasicTrainer:
    def __init__(self, lr=1e-2):
        self.learning_rate = lr
        self.optimizers = []

    def setup_function(self, task, args):
        for key in ("batch", "iterator"):
            if key not in args:
                continue
            options = {"random": iterate_random, "intact": iterate_ordered}
            iterator = options[task.config("order", "random")]
            args[key] = iterator(args[key], task.config("batch_size", 32))
        return task, args

    def setup_component(self, params):
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)
        self.optimizers.append(optimizer)
        return optimizer

    def prepare(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def run(self, context):
        task, args = context
        args = args.copy()
        if "batch" in args:
            args["batch"] = next(args["batch"])

        loss = task.function(**args)
        loss.backward()
        return loss.item()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def save(self, components):
        log = []
        for instance in components:
            if not hasattr(instance, "parameters"):
                continue

            cls = instance.__class__
            data = (cls.__module__ + "." + cls.__qualname__).encode("utf-8")
            digest = hashlib.blake2b(data, digest_size=8).hexdigest()
            filename = f"{cls.__name__}-{digest}.pkl"
            torch.save(instance, f"models/{filename}")
            log.append(cls.__qualname__)

        print("💾  Saved model snapshot for: {}.".format(", ".join(log)))
