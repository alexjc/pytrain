# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import os
import sys
import inspect
import importlib


class Function:
    def __init__(self, name, function, signature):
        self.name = name
        self.function = function
        self.signature = signature

    def config(self, key, default):
        return self.function._pytrain.get(key, default)


class Registry:
    def __init__(self, config):
        self.functions = []
        self.components = set()
        self.datasets = set()

        self.config = config

    def create_components(self):
        return {cp: cp() for cp in self.components}

    def create_datasets(self):
        return {ds: ds() for ds in self.datasets}

    def load(self):
        for root, _, files in os.walk(self.config.get("-r") or "train"):
            if root.endswith("__pycache__"):
                continue

            for filename in files:
                if not filename.startswith(self.config.get("-k") or "train_"):
                    continue

                module = self.import_module(os.path.join(root, filename))
                self.load_module(module)

    def import_module(self, path):
        name = path.replace(".py", "").replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def load_module(self, module):
        module_name = module.__name__.split(".", maxsplit=1)[1]
        for name in dir(module):
            if not name.startswith("task_"):
                continue

            function = getattr(module, name)
            vars(function).setdefault("_pytrain", {})

            qualname = module_name + "." + name
            signature = inspect.signature(function)
            self.functions.append(Function(qualname, function, signature))

            for param in signature.parameters.values():
                type_ = param.annotation
                if type_ == param.empty:
                    continue
                if hasattr(type_, "parameters"):
                    self.components.add(type_)
                if param.name in ("batch", "data", "iterator"):
                    self.datasets.add(type_)
