# PyTrain — Copyright (c) 2019, Alex J. Champandard.

__all__ = ["optimize_until"]


def optimize_until(iterations: int = -1):
    def wrapper(function):
        function._pytrain = {}
        function._pytrain["iterations"] = iterations
        return function

    return wrapper
