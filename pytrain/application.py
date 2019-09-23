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
from prompt_toolkit.utils import _CHAR_SIZES_CACHE

from . import __version__
from .trainer import BasicTrainer


class ShowBar(formatters.Formatter):
    template = "<bar>{start}<bar-a>{bar_a}</bar-a><bar-b>{bar_b}</bar-b><bar-c>{bar_c}</bar-c>{end}</bar>"

    def __init__(
        self, losses, start="[", end="]", sym_a="=", sym_b=">", sym_c=" ", unknown="#"
    ):
        assert len(sym_a) == 1 and formatters.get_cwidth(sym_a) == 1
        assert len(sym_c) == 1 and formatters.get_cwidth(sym_c) == 1

        self.losses = losses

        # Predictions for the size of Emoji is one off, critical feature.
        _CHAR_SIZES_CACHE["🚃"] = 1

        self.start = start
        self.end = end
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.sym_c = sym_c
        self.unknown = unknown

    def format(self, progress_bar, progress, width):
        if progress in self.losses:
            loss = self.losses[progress]
            return f"error={loss:1.2e}"

        width -= formatters.get_cwidth(self.start + self.sym_b + self.end)
        assert progress.total

        pb_a = int(progress.percentage * width / 100)
        bar_a = self.sym_a * pb_a
        bar_b = self.sym_b
        bar_c = self.sym_c * (width - pb_a)

        return HTML(self.template).format(
            start=self.start, end=self.end, bar_a=bar_a, bar_b=bar_b, bar_c=bar_c
        )

    def get_width(self, progress_bar):
        return formatters.D(min=9)


SCREEN_BANNER = HTML("<banner><b>PyTrain {}</b> - {}</banner>")
SCREEN_TOOLBAR = HTML("<b>[Control-L]</b> clear  <b>[Control-X]</b> quit")
SCREEN_STYLE = Style.from_dict(
    {"bottom-toolbar": "fg:cyan", "banner": "fg:cyan", "title": "fg:white"}
)
SCREEN_FORMATTERS = [
    formatters.Label(),
    # formatters.Text(" i="),
    # formatters.Progress(),
    formatters.Text(" "),
    "ShowBar",
    formatters.Text(" ETA ", style="class:time-left"),
    formatters.TimeLeft(),
]


class Application:
    def __init__(self, loop, registry):
        self.loop = loop
        self.registry = registry
        self.losses = {}

        self._components = self.registry.create_components()
        self._datasets = self.registry.create_datasets()
        self._tasks = []
        self.quit = False

    def prepare_fuction(self, function):
        args = {}
        for param in function.signature.parameters.values():
            type_ = param.annotation
            if type_ in self._components:
                args[param.name] = self._components[type_]
            if type_ in self._datasets:
                args[param.name] = self._datasets[type_]
        return args

    async def run_function(self, function):
        args = self.prepare_fuction(function)
        context = self.trainer.setup_function(function, args)

        progress = self.progress_bar(
            range(function.config("iterations", 100)),
            label="  - " + function.name,
            remove_when_done=True,
        )
        for i in progress:
            loss = self.trainer.run(context)
            self.losses[progress] = loss
            yield i

        print(f"📉  {function.name} approximate error={loss:1.2e}")
        await asyncio.sleep(0.01)

    async def run_components(self, components):
        start = time.time()

        params, label = [], []
        for cp in components:
            component = self._components[cp]
            label.append(
                component.__class__.__module__ + "." + component.__class__.__name__
            )
            params.extend(component.parameters())

        self.trainer.setup_component(params)
        progress = self.progress_bar(
            range(100), label=" ".join(label), remove_when_done=True
        )
        for i in progress:
            yield i

        elapsed = time.time() - start
        print(
            f"{' '.join(label)}\n🏁  Training completed in {elapsed:1.1f}s total time."
        )

        self.trainer.save([self._components[cp] for cp in components])
        await asyncio.sleep(0.01)

    def stop(self, _):
        self.quit = True

    async def main(self):
        bindings = KeyBindings()
        bindings.add("c-x")(self.stop)

        description = (
            f"Running {len(self.registry.functions)} task(s), "
            + f"optimizing {len(self.registry.components)} component(s)."
        )

        formatters = SCREEN_FORMATTERS.copy()
        formatters[formatters.index("ShowBar")] = ShowBar(
            self.losses, sym_a="_", sym_b="🚃 ", sym_c="․"
        )

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

        self.trainer = BasicTrainer()

        with self.progress_bar:
            self.progress_bar.title = HTML(f"<b>Stage 1</b>: {description}")

            for components, functions in self.registry.groups():
                task = self.run_components(components)
                self._tasks.append(task)

                for function in functions:
                    task = self.run_function(function)
                    self._tasks.append(task)

            while not self.quit and len(self._tasks):
                self.trainer.prepare()
                for task in list(self._tasks):
                    try:
                        await task.__anext__()
                    except StopAsyncIteration:
                        self._tasks.remove(task)

                self.trainer.step()

    def run(self):
        if len(self.registry.functions) == 0:
            print(f"ERROR: No tasks found in specified directory.")
            return

        os.makedirs("models", exist_ok=True)

        with patch_stdout():
            self.loop.run_until_complete(self.main())
