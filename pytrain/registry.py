# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import os
import sys
import inspect
import importlib
import collections


class Function:
    def __init__(self, name, function, signature):
        self.name = name
        self.function = function
        self.signature = signature

    def config(self, key, default):
        return self.function._pytrain.get(key, default)

    def dependencies(self):
        dependencies = []
        for param in self.signature.parameters.values():
            type_ = param.annotation
            if type_ == param.empty:
                continue
            if hasattr(type_, "parameters"):
                dependencies.append(type_)
        return tuple(dependencies)


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
        name = os.path.split(path)[1].replace(".py", "").replace("/", ".")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def load_module(self, module):
        module_name = module.__name__.split(".")[-1]
        for name in dir(module):
            if not name.startswith("task_"):
                continue

            function = getattr(module, name)
            config = vars(function).setdefault("_pytrain", {})

            qualname = module_name + "." + name
            signature = inspect.signature(function)
            self.functions.append(Function(qualname, function, signature))

            for param in signature.parameters.values():
                type_ = param.annotation
                if type_ == param.empty:
                    continue
                if hasattr(type_, "parameters"):
                    self.configure_component(type_, config)
                    self.components.add(type_)
                if param.name in ("batch", "data", "iterator"):
                    self.datasets.add(type_)

    def configure_component(self, type_, config):
        cfg = getattr(type_, "_pytrain", {})
        if "iteration" in config:
            cfg["iteration"] = max(config["iteration"], cfg.get("iteration", 0))
        setattr(type_, "_pytrain", cfg)

    def groups(self):
        groups = collections.defaultdict(list)
        for f in self.functions:
            groups[f.dependencies()].append(f)

        return groups.items()
