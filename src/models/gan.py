import os

from lightning import LightningModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import ops


class Generator(nn.Module):
    """
    Args:
        image_shape: (channels, width, height)
    """
    def __init__(self, latent_dim, image_shape: tuple[int]):
        super().__init__()
        self.image_shape = image_shape

        def block(in_features, out_features, normalize=True):
            layers = [nn.Linear(in_features, out_features)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_shape))),  # 784
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor):
        z = self.net(z)
        z = z.view(z.size(0), *self.image_shape)
        return z
    

class Discriminator(nn.Module):
    """
    Args:
        image_shape (`tuple(int)`): (channels, width, height)
    """
    def __init__(self, image_shape: tuple[int]):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = self.net(x)
        return x


class GanModel(LightningModule):
    """
    Generative Adversarial Network
    https://github.com/goodfeli/adversarial

    Args:
        channels (`int`): number of channels in image
        width (`int`): width of image
        height (`int`): height of image
        latent_dim (`int` default to 100): dimension of latent space
        lr (`float` default to 0.0002): learning rate
        b1 (`float` default to 0.5): beta 1
        b2 (`float` default to 0.999): beta 2
    """
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        data_shape = (channels, width, height)
        self.g = Generator(latent_dim=self.hparams.latent_dim, image_shape=data_shape)
        self.d = Discriminator(image_shape=data_shape)

        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)
        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(b1, b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch):
        images, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # sample noise
        z = torch.randn(images.size(0), self.hparams.latent_dim)
        z = z.type_as(images)

        # train generator
        # generate images
        self.toggle_optimizer(optimizer_g)
        self.generated_imgs = self(z)

        # log sampled images
        sample_imgs = self.generated_imgs[:6]
        self.training_step_outputs.append(sample_imgs)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image("generated_images", grid, 0)

        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)

        # adversarial loss is binary cross-entropy
        g_loss = F.binary_cross_entropy(self.discriminator(self(z)), valid)
        self.log("g_loss", g_loss, prog_bar=True)
        self.manual_backward(g_loss)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # train discriminator
        # Measure discriminator's ability to classify real from generated samples
        self.toggle_optimizer(optimizer_d)

        # how well can it label as real?
        valid = torch.ones(images.size(0), 1)
        valid = valid.type_as(images)
        real_loss = F.binary_cross_entropy(self.discriminator(imgs), valid)

        # how well can it label as fake?
        fake = torch.zeros(images.size(0), 1)
        fake = fake.type_as(images)
        fake_loss = F.binary_cross_entropy(self.discriminator(self(z).detach()), fake)

        # discriminator loss is the average of these
        d_loss = (real_loss + fake_loss) / 2
        self.log("d_loss", d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

    def on_validation_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)
        # log sampled images
        sample_imgs = self(z)
        # self.validation_step_outputs.append(sample_imgs)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.log("generated_images", grid, self.current_epoch)
