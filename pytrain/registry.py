# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import os
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
    def __init__(self):
        self.functions = []
        self.modules = []

    def load(self):
        for root, _, files in os.walk("examples"):
            if root.endswith("__pycache__"):
                continue

            for filename in files:
                if not filename.startswith("train_"):
                    continue

                basename = os.path.splitext(filename)[0]
                module = importlib.import_module(
                    os.path.join(root, basename).replace("/", ".")
                )
                self.load_module(module)

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
                    self.modules.append(module)
