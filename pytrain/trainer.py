# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import hashlib
import itertools

import torch

from .data import Batch


def iterate_ordered(data, batch_size):
    for i in itertools.count():
        indices = torch.arange(i * batch_size, (i + 1) * batch_size, step=+1)
        b = Batch.from_data(data[indices % len(data)])
        b.number = i
        yield b


def iterate_random(data, batch_size):
    while True:
        indices = torch.randint(0, len(data), size=(batch_size,))
        yield Batch.from_data(data[indices])


def get_config(obj, name, default):
    return getattr(obj, "_pytrain", {}).get(name, default)


class BasicTrainer:
    def __init__(self, device, lr=1e-2):
        self.device = device
        self.learning_rate = lr
        self.samples = None
        self.optimizers = []
        self.schedulers = []

    def setup_function(self, function, args, mode):
        for key in args.keys():
            if isinstance(args[key], torch.nn.Module):
                if mode == "training":
                    args[key].train()
                if mode == "validation":
                    args[key].eval()

            if key.split("_")[0] not in ("batch", "iterator"):
                continue
            options = {"training": iterate_random, "validation": iterate_ordered}
            iterator = options[function.config("order", None) or mode]
            args[key] = iterator(args[key], function.config("batch_size", 32))
        return function, args

    def setup_components(self, components):
        opt_class = torch.optim.Adam
        sch_class = torch.optim.lr_scheduler.CyclicLR

        all_params, lr = [], self.learning_rate
        for cp in components:
            params = list(cp.parameters())
            if all(p.requires_grad == False for p in params):
                print("WARNING:", cp, id(cp), "requires no gradients.")
                continue

            all_params.extend(params)
            lr = get_config(cp, "learning_rate", lr)
            opt_class = get_config(cp, "optimizer_class", opt_class)
            sch_class = get_config(cp, "scheduler_class", sch_class)

        optimizer = opt_class(all_params, lr=lr)
        scheduler = sch_class(
            optimizer,
            base_lr=lr * 1e-2,
            max_lr=lr,
            mode="triangular",
            step_size_up=100,
            step_size_down=200,
            cycle_momentum=False,
        )
        self.optimizers.append(optimizer)
        self.schedulers.append(scheduler)
        return optimizer

    def prepare(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        self.samples = 0

    def run_training(self, context):
        task, args = context
        args = args.copy()
        for key in args:
            if key.split("_")[0] == "batch":
                args[key] = next(args[key]).to(self.device)

        self.samples += 1
        loss = task.function(**args)
        if isinstance(loss, float):
            return loss
        else:
            loss.backward()
            return loss.item()

    def run_validation(self, context):
        task, args = context
        args = args.copy()
        for key in args:
            if key.split("_")[0] == "batch":
                args[key] = next(args[key]).to(self.device)

        with torch.no_grad():
            score = task.function(**args)
        return score

    def report(self, loss):
        pass

    def step(self):
        if self.samples == 0:
            return

        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()

    def save(self, components):
        log = []
        for instance in components:
            if not hasattr(instance, "parameters"):
                continue

            cls = instance.__class__
            data = (cls.__module__ + "." + cls.__qualname__).encode("utf-8")
            digest = hashlib.blake2b(data, digest_size=8).hexdigest()
            filename = f"{cls.__name__}-{digest}.pkl"
            torch.save(instance.state_dict(), f"models/{filename}")
            log.append(cls.__qualname__)

        print("💾  Saved model snapshot for: {}.".format(", ".join(log)))
