# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import asyncio

from .application import Application
from .registry import Registry


def main():
    registry = Registry()
    registry.load()

    loop = asyncio.get_event_loop()
    application = Application(loop, registry)
    application.run()


if __name__ == "__main__":
    main()
