from typing import Any, Optional
import lightning.pytorch as pl
import wandb
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim
from torchvision.transforms.functional import to_pil_image
from lightning.pytorch.callbacks import Callback
# from lightgan.utils import logger
import torchvision
from torchvision import transforms


class InverseNormalize(transforms.Normalize):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class LogPredictionSamples(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # sample_imgs = pl_module.training_step_outputs[0].detach()
        sample_imgs = pl_module.training_step_outputs[0]
        sample_imgs2 = pl_module.training_step_outputs2[0]
        # sample_imgs = pl_module.training_step_outputs[:6]
        grid = torchvision.utils.make_grid(sample_imgs, 4)
        grid = InverseNormalize()(grid)
        grid = to_pil_image(grid)
        grid2 = torchvision.utils.make_grid(sample_imgs2, 4)
        grid2 = InverseNormalize()(grid2)
        grid2 = to_pil_image(grid2)
        trainer.logger.log_table(
            key="training samples",
            columns=["real", "fake"],
            data=[[wandb.Image(grid), wandb.Image(grid2)]]
        )
        pl_module.training_step_outputs.clear()
        pl_module.training_step_outputs2.clear()

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     sample_imgs = pl_module.validation_step_outputs
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     trainer.logger.experiment.log(
    #         {"validation examples": [wandb.Image(grid)]}
    #     )
    #     pl_module.validation_step_outputs.clear()
