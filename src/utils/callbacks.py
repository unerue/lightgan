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
    """
    Args:
        list[torch.Tensor], 0-255, shape (B, C, H, W)
    """
    def on_train_epoch_end(self, trainer, pl_module):
        if isinstance(pl_module.training_step_outputs1, list):
            real_samples = torch.cat(pl_module.training_step_outputs1, dim=0)
            fake_samples = torch.cat(pl_module.training_step_outputs2, dim=0)
        
        real_samples = torchvision.utils.make_grid(real_samples[:8], 4)
        real_samples = InverseNormalize()(real_samples)
        real_samples = to_pil_image(real_samples)

        fake_samples = torchvision.utils.make_grid(fake_samples[:8], 4)
        fake_samples = InverseNormalize()(fake_samples)
        fake_samples = to_pil_image(fake_samples)
        trainer.logger.log_table(
            key="training samples",
            columns=["real", "fake"],
            data=[[wandb.Image(real_samples), wandb.Image(fake_samples)]]
        )
        pl_module.training_step_outputs1.clear()
        pl_module.training_step_outputs2.clear()

    # def on_validation_epoch_end(self, trainer, pl_module):
    #     sample_imgs = pl_module.validation_step_outputs
    #     grid = torchvision.utils.make_grid(sample_imgs)
    #     trainer.logger.experiment.log(
    #         {"validation examples": [wandb.Image(grid)]}
    #     )
    #     pl_module.validation_step_outputs.clear()
