# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import os
import sys
import inspect
import importlib
import collections

from .data import Dataset


def is_component(param):
    """Rules of thumb to guess if a type annotation can be considered a component.
    """
    type_ = param.annotation
    return hasattr(type_, "parameters") or callable(type_)


def is_dataset(param):
    """Rules of thumb to guess if a type annotation can be considered a dataset.
    """
    return param.name.split("_")[0] in ("batch", "data", "iterator")


class Function:
    def __init__(self, name, function, signature):
        self.name = name
        self.function = function
        self.signature = signature

    @classmethod
    def from_callable(self, module_name, function):
        qualname = module_name + "." + function.__name__
        signature = inspect.signature(function)
        return Function(qualname, function, signature)

    def config(self, key, default):
        return self.function._pytrain.get(key, default)

    def dependencies(self):
        dependencies = []
        for param in self.signature.parameters.values():
            type_ = param.annotation
            if type_ == param.empty:
                continue
            if not is_dataset(param) and is_component(param):
                dependencies.append(type_)
        return tuple(dependencies)


class Registry:
    def __init__(self, config):
        self.functions = []
        self.components = set()
        self.datasets = set()

        self.config = config

    def create_components(self, device):
        def _create(cp):
            try:
                return cp(pretrained=self.config.get("--resume"))
            except TypeError:
                return cp()

        return {cp: _create(cp).to(device) for cp in self.components}

    def create_datasets(self):
        return {ds: Dataset.from_data(ds()) for ds in self.datasets}

    def load(self):
        sys.path.append(os.getcwd())

        for root, _, files in os.walk(self.config.get("--path") or "train"):
            if root.endswith("__pycache__"):
                continue

            for filename in files:
                if filename.startswith("_") or not filename.endswith(".py"):
                    continue
                if (self.config.get("--include") or "train_") not in filename:
                    continue

                module = self.import_module(os.path.join(root, filename))
                self.load_module(module)

    def import_module(self, path):
        name = os.path.split(path)[1].replace(".py", "").replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def load_module(self, module):
        module_name = module.__name__.split(".")[-1]
        for name in dir(module):
            if not name.split("_")[0] in ("task", "show", "main"):
                continue

            obj = getattr(module, name)
            config = vars(obj).setdefault("_pytrain", {})

            function = Function.from_callable(module_name, obj)
            self.functions.append(function)

            for param in function.signature.parameters.values():
                type_ = param.annotation
                if type_ == param.empty:
                    continue
                if is_dataset(param):
                    self.datasets.add(type_)
                elif is_component(param):
                    self.configure_component(type_, config)
                    self.components.add(type_)

    def configure_component(self, type_, config):
        cfg = getattr(type_, "_pytrain", {})
        if "iteration" in config:
            cfg["iteration"] = max(config["iteration"], cfg.get("iteration", 0))
        setattr(type_, "_pytrain", cfg)

    def groups(self):
        groups = collections.defaultdict(list)
        for f in self.functions:
            groups[f.dependencies()].append(f)

        # Strict subsets of other groups are merged into parent.
        for group in list(groups.keys()):
            for g in list(groups.keys()):
                if set(g) < set(group):
                    groups[group].extend(groups[g])
                    del groups[g]

        return groups.items()
