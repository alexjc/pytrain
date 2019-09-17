# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import asyncio
import traceback

from prompt_toolkit import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts.progress_bar import formatters

from . import __version__
from .trainer import BasicTrainer


SCREEN_TITLE = HTML("<b>PyTrain {}</b> — {}")
SCREEN_TOOLBAR = HTML("<b>[Control-L]</b> clear  <b>[Control-X]</b> quit")
SCREEN_STYLE = Style.from_dict({"bottom-toolbar": "fg:cyan", "title": "fg:cyan"})
SCREEN_FORMATTERS = [
    formatters.Label(),
    formatters.Text(" "),
    formatters.Progress(),
    formatters.Text(" "),
    formatters.Text("ETA ", style="class:time-left"),
    formatters.TimeLeft(),
    formatters.Text("", style="class:time-left"),
    formatters.Text(" "),
    formatters.Bar(sym_a="_", sym_b="🚃 ", sym_c="․"),
]


class Application:
    def __init__(self, loop, registry):
        self.loop = loop
        self.registry = registry

        self._components = self.registry.create_instances()
        self._tasks = []

    def prepare_task(self, task):
        args, parameters = [], []
        for param in task.signature.parameters.values():
            argument = param.annotation
            if argument in self._components:
                module = self._components[argument]
                args.append(module)
                parameters.extend(module.parameters())
            else:
                args.append(argument())
        return args, parameters

    async def run_task(self, task):
        try:
            args, params = self.prepare_task(task)

            trainer = BasicTrainer(params)
            for _ in self.progress_bar(
                range(task.config("iterations", 100)),
                label=task.name,
                remove_when_done=True,
            ):
                trainer.step(task, args)

            trainer.save(args)
            await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            pass
        except:
            traceback.print_exc()

    def stop(self, _):
        for task in self._tasks:
            task.cancel()

    async def main(self):
        bindings = KeyBindings()
        bindings.add("c-x")(self.stop)

        description = (
            f"Running {len(self.registry.functions)} task(s), "
            + f"optimizing {len(self.registry.components)} component(s)."
        )

        self.progress_bar = ProgressBar(
            title=SCREEN_TITLE.format(__version__, description),
            bottom_toolbar=SCREEN_TOOLBAR,
            style=SCREEN_STYLE,
            key_bindings=bindings,
            formatters=SCREEN_FORMATTERS,
        )

        with self.progress_bar:
            for function in self.registry.functions:
                task = self.loop.create_task(self.run_task(function))
                self._tasks.append(task)

            if len(self._tasks) > 0:
                await asyncio.wait(self._tasks)

    def run(self):
        if len(self.registry.functions) == 0:
            print(f"ERROR: No tasks found in specified directory.")
            return

        with patch_stdout():
            self.loop.run_until_complete(self.main())
