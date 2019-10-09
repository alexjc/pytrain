#!/usr/bin/env python3
# PyTrain — Copyright (c) 2019, Alex J. Champandard.
"""
Usage: pytrain [-k FILTER] [-r ROOTDIR] [-d DEVICE]

Options:
  -k FILTER         Select which tests to run by matching this substring filter.
  -r ROOTDIR        Base directory from which to collect all the tasks.
  -d DEVICE --device DEVICE   Device to use for optimizing the components. [default: cpu]
"""

import asyncio

from docopt import docopt

from . import __version__
from .application import Application
from .registry import Registry


def main():
    config = docopt(__doc__, version=f"pytrain {__version__}")

    registry = Registry(config)
    registry.load()

    loop = asyncio.get_event_loop()
    application = Application(loop, config["--device"], registry)
    application.run()


if __name__ == "__main__":
    main()
