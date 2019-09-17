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

        self.config = config

    def create_instances(self):
        return {cp: cp() for cp in self.components}

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
        for name in dir(module):
            if not name.startswith("task_"):
                continue

            function = getattr(module, name)
            if not hasattr(function, "_pytrain"):
                function._pytrain = {}

            signature = inspect.signature(function)
            self.functions.append(Function(name, function, signature))

            for param in signature.parameters.values():
                module = param.annotation
                if module == param.empty:
                    continue
                if hasattr(module, "parameters"):
                    self.components.add(module)
