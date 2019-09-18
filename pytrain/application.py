# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import os
import time
import asyncio
import traceback

from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts.progress_bar import formatters

from . import __version__
from .trainer import BasicTrainer


SCREEN_BANNER = HTML("<banner><b>PyTrain {}</b> - {}</banner>")
SCREEN_TOOLBAR = HTML("<b>[Control-L]</b> clear  <b>[Control-X]</b> quit")
SCREEN_STYLE = Style.from_dict(
    {"bottom-toolbar": "fg:cyan", "banner": "fg:cyan", "title": "fg:white"}
)
SCREEN_FORMATTERS = [
    formatters.Label(),
    formatters.Text(" e="),
    None,
    formatters.Text(" i="),
    formatters.Progress(),
    formatters.Text(" "),
    formatters.Text("ETA ", style="class:time-left"),
    formatters.TimeLeft(),
    formatters.Text("", style="class:time-left"),
    formatters.Text(" "),
    formatters.Bar(sym_a="_", sym_b="🚃 ", sym_c="․"),
]


class ShowLoss(formatters.Formatter):
    def __init__(self, losses: dict):
        self.losses = losses

    def format(self, progress_bar, progress, width):
        loss = self.losses.get(progress, 0.0)
        return f"{loss:1.2e}"

    def get_width(self, progress_bar):
        return formatters.D.exact(8)


class Application:
    def __init__(self, loop, registry):
        self.loop = loop
        self.registry = registry
        self.losses = {}

        self._components = self.registry.create_components()
        self._datasets = self.registry.create_datasets()
        self._tasks = []

    def prepare_task(self, task):
        args, model = {}, []
        for param in task.signature.parameters.values():
            type_ = param.annotation
            if type_ in self._components:
                component = self._components[type_]
                args[param.name] = component
                model.extend(component.parameters())
            if type_ in self._datasets:
                dataset = self._datasets[type_]
                args[param.name] = dataset
        return args, model

    async def run_task(self, task):
        try:
            start = time.time()
            args, params = self.prepare_task(task)

            trainer = BasicTrainer(task, args, params)
            progress = self.progress_bar(
                range(task.config("iterations", 100)),
                label=task.name,
                remove_when_done=True,
            )
            for _ in progress:
                loss = trainer.step()
                self.losses[progress] = loss
                await asyncio.sleep(0.0)

            elapsed = time.time() - start
            print(f"{task.name}\n🏁  Task completed in {elapsed:1.1f}s total time.")
            print(f"📉  Approximate training error={loss:1.2e}")
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

        formatters = SCREEN_FORMATTERS.copy()
        formatters[formatters.index(None)] = ShowLoss(self.losses)

        project = os.path.basename(os.getcwd())
        print_formatted_text(
            SCREEN_BANNER.format(__version__, project), style=SCREEN_STYLE
        )

        self.progress_bar = ProgressBar(
            bottom_toolbar=SCREEN_TOOLBAR,
            style=SCREEN_STYLE,
            key_bindings=bindings,
            formatters=formatters,
        )

        with self.progress_bar:
            self.progress_bar.title = HTML(f"<b>Stage 1</b>: {description}")

            for function in self.registry.functions:
                task = self.loop.create_task(self.run_task(function))
                self._tasks.append(task)

            if len(self._tasks) > 0:
                await asyncio.wait(self._tasks)

    def run(self):
        if len(self.registry.functions) == 0:
            print(f"ERROR: No tasks found in specified directory.")
            return

        os.makedirs("models", exist_ok=True)

        with patch_stdout():
            self.loop.run_until_complete(self.main())
