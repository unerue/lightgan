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
    def __init__(self, nz: int = 100, ngf: int = 64):
        super().__init__()

        def block(in_channels, out_channels, kernel_size, stride, padding, normalize=True):
            layers = [
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, padding=padding, bias=False
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            *block(nz, ngf * 8, 4, 1, 0),
            *block(ngf * 8, ngf * 4, 4, 2, 1),
            *block(ngf * 4, ngf * 2, 4, 2, 1),
            *block(ngf * 2, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = self.model(z)
        return z
        

class Discriminator(nn.Module):
    def __init__(self, ndf: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, images):
        images = self.model(images)
        return images.view(-1, 1).squeeze(1)
    

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DcGanModel(LightningModule):
    def __init__(
        self,
        channels,
        width,
        height,
        latent_dim: int = 100,
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        batch_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        data_shape = (channels, width, height)
        self.generator = Generator()
        self.generator.apply(initialize_weights)
        self.discriminator = Discriminator()
        self.discriminator.apply(initialize_weights)

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

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def training_step(self, batch):
        imgs, _ = batch

        optimizer_g, optimizer_d = self.optimizers()

        # train discriminator
        self.toggle_optimizer(optimizer_d)
        # train with real
        # real_label = 1
        label = torch.full((imgs.size(0),), 1, dtype=torch.float, device=self.device)
        output = self.discriminator(imgs)
        d_loss_real = F.binary_cross_entropy(output, label)

        self.manual_backward(d_loss_real)

        d_x = output.mean().item()

        # train with fake
        # fake_label = 0
        noise = torch.randn(imgs.size(0), 100, 1, 1, device=self.device)
        fake = self.generator(noise)
        label.fill_(0)

        output = self.discriminator(fake.detach())
        d_fake_loss = F.binary_cross_entropy(output, label)
        self.manual_backward(d_fake_loss)
        d_g_z1 = output.mean().item()
        err_d = d_loss_real + d_fake_loss

        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # train generators
        self.toggle_optimizer(optimizer_g)
        label.fill_(1)
        output = self.discriminator(fake)
        g_loss = F.binary_cross_entropy(output, label)
        self.manual_backward(g_loss)
        d_g_z2 = output.mean().item()
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        fakes = self.generator(noise)
        self.training_step_outputs.append(fakes)
