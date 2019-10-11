# PyTrain — Copyright (c) 2019, Alex J. Champandard.

__all__ = ["terminates", "iterates"]


def _annotate(obj, **kwargs):
    if not hasattr(obj, "_pytrain"):
        config = {}
        setattr(obj, "_pytrain", config)
    else:
        config = getattr(obj, "_pytrain")

    for key, value in kwargs.items():
        if value is not None:
            config[key] = value
    return obj


def iterates(batch_size: int = None, order: int = None):
    def wrapper(function):
        return _annotate(function, batch_size=batch_size, order=order)

    return wrapper


def terminates(
    component=None, iteration: int = None, epoch: int = None, threshold: float = None
):
    if component is None:

        def wrapper(function):
            return _annotate(function, iteration=iteration, threshold=threshold)

        return wrapper

    _annotate(component, iteration=iteration, epoch=epoch, threshold=threshold)
