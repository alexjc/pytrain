# PyTrain — Copyright (c) 2019, Alex J. Champandard.

import torch
import pytrain


class MyImage(torch.nn.Module):
    """Stores and returns a square tensor that can be optimized.
    """

    def __init__(self):
        super(MyImage, self).__init__()

        image = torch.zeros((512, 512), dtype=torch.float)
        self.data = torch.nn.Parameter(image, requires_grad=True)

    def forward(self):
        return self.data


class MyData:
    """Simple iterator that returns a single randomly-chosen target image.
    """

    def __init__(self):
        self.target = torch.empty((512, 512), dtype=torch.float).uniform_()

    def __next__(self):
        yield pytrain.Batch(target=self.target)


@pytrain.optimize_until(iterations=1000)
def task_optimize_image(data: MyData, module: MyImage):
    output = module()
    return torch.nn.functional.mse_loss(output, data.target)
