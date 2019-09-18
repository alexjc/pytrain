# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import hashlib

import torch


class BasicTrainer:
    def __init__(self, parameters, lr=1e-2):
        self.optimizer = torch.optim.Adam(parameters, lr=lr)

    def step(self, task, args):
        self.optimizer.zero_grad()
        args = args.copy()
        if "batch" in args:
            args["batch"] = next(args["batch"])
        loss = task.function(**args)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, args):
        for instance in args.values():
            if not hasattr(instance, "parameters"):
                continue

            cls = instance.__class__
            data = (cls.__module__ + "." + cls.__qualname__).encode("utf-8")
            digest = hashlib.blake2b(data, digest_size=8).hexdigest()
            filename = f"{cls.__name__}-{digest}.pkl"
            torch.save(instance, f"models/{filename}")
            print(f"💾 ", cls.__module__, filename)
