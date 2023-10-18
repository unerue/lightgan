import os
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from lightning import LightningModule

from . import networks


class BaseModel(ABC, LightningModule):
    """
    This class is an abstract base class (ABC) for GAN models.
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    @abstractmethod
    def generator_training_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def discriminator_training_step(self, batch, batch_idx):
        ...
