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

    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals

            return grad_hook

        return hook_gen, saved_dict

    @classmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def generator_training_step(self, batch):
        ...

    @abstractmethod
    def discriptor_training_step(self, batch):
        ...
