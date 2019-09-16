# PyTrain — Copyright (c) 2019, Alex J. Champandard.
"""
Usage: pytrain [-k FILTER] [-r ROOTDIR]

Options:
  -k FILTER         Select which tests to run by matching this substring filter.
  -r ROOTDIR        Base directory from which to collect all the tasks.
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
    application = Application(loop, registry)
    application.run()


if __name__ == "__main__":
    main()
