# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import os
import sys
import time
import asyncio
import itertools

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
        if sys.platform == "darwin":
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
            return f"error={loss:1.4e}"

        width -= formatters.get_cwidth(self.start + self.sym_b + self.end)

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
    formatters.Text(" "),
    "ShowBar",
    formatters.Text(" ETA ", style="class:time-left"),
    formatters.TimeLeft(),
]


class Application:
    def __init__(self, loop, device, registry):
        self.loop = loop
        self.device = device
        self.registry = registry
        self.losses = {}

        self._components = self.registry.create_components(self.device)
        self._datasets = self.registry.create_datasets()
        self._tasks = []
        self.quit = False

    def prepare_function(self, function, mode="training"):
        args, length = {}, None
        for param in function.signature.parameters.values():
            type_ = param.annotation
            if type_ in self._components:
                args[param.name] = self._components[type_]
            if type_ in self._datasets:
                assert length is None, "Only one dataset per function supported."
                data = getattr(self._datasets[type_], mode)
                if data is None:
                    length = -1
                else:
                    length = len(data) // function.config("batch_size", 32)
                args[param.name] = data
        return args, length

    async def run_function(self, function, args, iterations, mode):
        context = self.trainer.setup_function(function, args, mode)
        run_one_batch = getattr(self.trainer, "run_" + mode)

        progress = self.progress_bar(
            data=range(iterations), label="  - " + function.name, remove_when_done=True
        )
        self.losses[progress] = float("+inf")
        for i in progress:
            loss = run_one_batch(context)
            if loss is None:
                break
            yield progress, loss

    def prepare_components(self, components):
        epochs = 1
        for cp in components:
            config = getattr(cp, "_pytrain", {})
            epochs = max(epochs, config.get("epoch", 0))
        return epochs

    async def run_all_functions(self, epoch, functions, mode="training"):
        args, length = [], 0
        for function in functions:
            a, l = self.prepare_function(function, mode)
            args.append(a)
            length = max(l, length)

        if length == 0:
            return

        children = [
            self.run_function(f, a, length, mode=mode) for f, a in zip(functions, args)
        ]
        total = 0.0
        for j in self.progress_bar(
            range(length + 1), label=mode, remove_when_done=True
        ):
            for task in list(children):
                try:
                    progress, loss = await task.__anext__()
                    total += loss
                except StopAsyncIteration:
                    children.remove(task)

                self.losses[progress] = total / (j + 1)

            if len(children) == 0:
                break

            yield j

        print(f"📉  {mode.capitalize()} loss for epoch #{epoch} is {total/length}.")

    async def run_components(self, components, functions, epochs):
        start = time.time()

        label, instances = [], []
        for component in components:
            cp = self._components[component]
            instances.append(cp)
            label.append(cp.__class__.__module__ + "." + cp.__class__.__name__)

        self.trainer.setup_components(instances)
        for i in self.progress_bar(
            range(epochs), label=" ".join(label), remove_when_done=True
        ):
            async for j in self.run_all_functions(i, functions, mode="training"):
                yield j

            async for j in self.run_all_functions(i, functions, mode="validation"):
                yield j

        elapsed = time.time() - start
        print(
            f"{' '.join(label)}\n"
            + f"🏁  Training completed in {elapsed:1.1f}s total time."
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

        self.trainer = BasicTrainer(device=self.device)

        with self.progress_bar:
            self.progress_bar.title = HTML(f"<b>Stage 1</b>: {description}")

            for components, functions in self.registry.groups():
                epoch = self.prepare_components(components)
                root = self.run_components(components, functions, epoch)
                self._tasks.append(root)

            while not self.quit and len(self._tasks) > 0:
                self.trainer.prepare()
                for root in list(self._tasks):
                    try:
                        await root.__anext__()
                    except StopAsyncIteration:
                        self._tasks.remove(root)

                self.trainer.step()

    def run(self):
        if len(self.registry.functions) == 0:
            print(f"ERROR: No tasks found in specified directory.")
            return

        os.makedirs("models", exist_ok=True)

        with patch_stdout():
            self.loop.run_until_complete(self.main())
