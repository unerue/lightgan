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


class LogPredictionSamples(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # sample_imgs = pl_module.training_step_outputs[0].detach()
        sample_imgs = pl_module.training_step_outputs[0]
        # sample_imgs = pl_module.training_step_outputs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        grid = to_pil_image(grid)
        trainer.logger.log_table(
            key="training samples",
            columns=["predictions"],
            data=[[wandb.Image(grid)]]
        )
        pl_module.training_step_outputs.clear()

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     sample_imgs = pl_module.validation_step_outputs
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     trainer.logger.experiment.log(
    #         {"validation examples": [wandb.Image(grid)]}
    #     )
    #     pl_module.validation_step_outputs.clear()
