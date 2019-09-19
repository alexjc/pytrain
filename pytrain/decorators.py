# PyTrain — Copyright (c) 2019, Alex J. Champandard.

__all__ = ["terminates", "iterates"]


def _annotate(function, **kwargs):
    config = vars(function).setdefault("_pytrain", {})
    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    return function


def terminates(iteration: int = None, threshold: float = None):
    def wrapper(function):
        return _annotate(function, iterations=iteration, threshold=threshold)

    return wrapper


def iterates(batch_size: int = None, order: int = None):
    def wrapper(function):
        return _annotate(function, batch_size=batch_size, order=order)

    return wrapper
