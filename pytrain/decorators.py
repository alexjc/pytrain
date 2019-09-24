# PyTrain — Copyright (c) 2019, Alex J. Champandard.

__all__ = ["terminates", "iterates"]


def _annotate(function, **kwargs):
    config = vars(function).setdefault("_pytrain", {})
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    return function


def iterates(batch_size: int = None, order: int = None):
    def wrapper(function):
        return _annotate(function, batch_size=batch_size, order=order)

    return wrapper


def terminates(component, iteration: int = None, threshold: float = None):
    if not hasattr(component, "_pytrain"):
        config = {}
        setattr(component, "_pytrain", config)
    else:
        config = getattr(component, "_pytrain")

    for key, value in dict(iterations=iteration, threshold=threshold).items():
        if value is not None:
            config[key] = value
