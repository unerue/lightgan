import itertools
from typing import Any, Optional

import torch
from torch import nn, optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from ..ops import ResnetGenerator, NLayerDiscriminator, ImagePool, PatchSampleF, GANLoss, PatchNCELoss


class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # resnet_9blocks
        self.x = ResnetGenerator(
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm_layer=nn.InstanceNorm2d,
            use_dropout=True,
            num_blocks=9,
            padding_type="reflect",
            no_antialias=True,
            no_antialias_up=True
        )


class F(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.x = PatchSampleF(
            use_mlp=True,
            init_type="normal",
            nc=256,
        )


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # basic_cond
        self.x = NLayerDiscriminator(
            input_nc=3,
            ndf=64,
            n_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=True,
        )


def initialize_weights(m, init_type="normal", init_gain=0.02):  # define the initialization function
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            nn.init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        nn.init.normal_(m.weight.data, 1.0, init_gain)
        nn.init.constant_(m.bias.data, 0.0)


def cyclegan_mse_loss(preds, label: str):
    if label.lower() == "real":
        target = torch.ones_like(preds)
    else:
        target = torch.zeros_like(preds)
    return F.mse_loss(preds, target)


def set_requires_grad(nets, requires_grad):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad


class CutModel(LightningModule):
    def __init__(
        self,
        optimizer1: optim.Optimizer,
        optimizer2: optim.Optimizer,
        scheduler1: Optional[Any] = None,
        scheduler2: Optional[Any] = None,
        image_size: int = 256,
        lambda_a: float = 10.0,
        lambda_b: float = 10.0,
        lambda_i: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lambda_a = 10.0
        self.g = Generator()
        self.d = Discriminator()

        self.g.x.apply(initialize_weights)
        self.g.y.apply(initialize_weights)
        self.d.x.apply(initialize_weights)
        self.d.y.apply(initialize_weights)

        self.fake_pool_a = ImagePool(50)
        self.fake_pool_b = ImagePool(50)

        criterion1 = GanLoss()
        criterion2 = PatchNCELoss()

        fid = FrechetInceptionDistance(feature=64)
        kid = KernelInceptionDistance(feature=64)

        image_shape = (3, 256, 256)
        self.example_input_array = (
            torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        )
        self.training_step_outputs = []
        self.training_step_outputs2 = []

    def forward(self, x, z):
        image_a, image_b = x, z
        return self.g.x(image_a), self.g.y(image_b)

    def configure_optimizers(self) -> Any:
        optimizer_g = self.hparams.optimizer1(
            itertools.chain(self.g.x.parameters(), self.g.y.parameters())
        )
        optimizer_d = self.hparams.optimizer2(
            itertools.chain(self.d.x.parameters(), self.d.y.parameters())
        )
        if self.hparams.scheduler1 is not None and self.hparams.scheduler2 is not None:
            scheduler_g = self.hparams.scheduler1(optimizer_g)
            scheduler_d = self.hparams.scheduler2(optimizer_d)
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

        return [optimizer_g, optimizer_d], []

    def generator_training_step(self, image_a, image_b):
        """Calculate GAN loss for the generator"""
        fake_b = self.g.x(image_a)
        cycle_a = self.g.y(fake_b)

        fake_a = self.g.y(image_b)
        cycle_b = self.g.x(fake_a)

        same_b = self.g.x(image_b)
        same_a = self.g.y(image_a)

        # generator genX must fool discrim disY so label is real = 1
        # GAN loss D_A(G_A(A))
        # print(self.d.x.requires_grad_)
        pred_fake_b = self.d.x(fake_b)
        gen_b_loss = cyclegan_mse_loss(pred_fake_b, "real")
        # self.log("gen_b_loss", gen_b_loss)

        # generator genY must fool discrim disX so label is real
        # GAN loss D_B(G_B(B))
        pred_fake_a = self.d.y(fake_a)
        gen_a_loss = cyclegan_mse_loss(pred_fake_a, "real")
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        identity_loss = F.l1_loss(same_a, image_a) + F.l1_loss(same_b, image_b)
        cycle_loss = F.l1_loss(cycle_a, image_a) + F.l1_loss(cycle_b, image_b)
        # self.log("cycle_loss", cycle_loss)
        # self.log("identity_loss", identity_loss)
        extra_loss = cycle_loss + 0.5 * identity_loss
        # self.log("extr0a_loss", extra_loss)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # Backward cycle loss || G_A(G_B(B)) - B||
        gen_loss = gen_a_loss + gen_b_loss + self.lambda_a * extra_loss
        self.log("gen_loss", gen_loss, prog_bar=True)
        self.fake_a = fake_a.detach()
        self.fake_b = fake_b.detach()

        return gen_loss

    def discriminator_training_step(self, image_a, image_b):
        """Calculate GAN loss for the discriminator

        Args:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
            We also call loss_D.backward() to calculate the gradients.
        """
        # Calculate GAN loss for discriminator D_A, D_B
        fake_a = self.fake_pool_a.query(self.fake_a)
        fake_b = self.fake_pool_b.query(self.fake_b)

        pred_real_a = self.d.x(image_b)
        real_a_loss = cyclegan_mse_loss(pred_real_a, "real")
        # self.log("real_a_loss", real_a_loss)
        pred_fake_a = self.d.x(fake_b)
        fake_a_loss = cyclegan_mse_loss(pred_fake_a, "fake")
        # self.log("fake_a_loss", fake_a_loss)

        pred_real_b = self.d.y(image_a)
        real_b_loss = cyclegan_mse_loss(pred_real_b, "real")
        # self.log("real_b_loss", real_b_loss)
        pred_fake_b = self.d.y(fake_a)
        fake_b_loss = cyclegan_mse_loss(pred_fake_b, "fake")
        # self.log("fake_b_loss", fake_b_loss)
        dis_loss = 0.5 * (real_a_loss + fake_a_loss + real_b_loss + fake_b_loss)
        self.log("dis_loss", dis_loss, prog_bar=True)
        return dis_loss

    def training_step(self, batch, batch_idx):
        image_a, image_b = batch
        optimizer_g, optimizer_d = self.optimizers()
        # scheduler_g, scheduler_d = self.lr_schedulers()

        # self.untoggle_optimizer(optimizer_d)
        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        g_loss = self.generator_training_step(image_a, image_b)
        self.manual_backward(g_loss)
        optimizer_g.step()
        # scheduler_g.step()
        self.untoggle_optimizer(optimizer_g)

        if len(self.training_step_outputs) < 7:
            fake_b = self.g.x(image_a).detach()
            fake_a = self.g.y(image_b).detach()

            # fake_a = self.g.y(image_b).detach()
            # reco_b = self.g.x(fake_a).detach()
            self.training_step_outputs.append(fake_b)
            self.training_step_outputs2.append(fake_a)

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        d_loss = self.discriminator_training_step(image_a, image_b)
        self.manual_backward(d_loss)
        optimizer_d.step()
        # scheduler_d.step()
        self.untoggle_optimizer(optimizer_d)
        # dis_required_grad = (optimizer_idx == 1)
        # set_requires_grad([self.dis.x, self.dis.y], dis_required_grad)
        # if optimizer_idx == 0:
        #     return self.generate_training_step(image_a, image_b)
        # else:
        #     return self.discriminator_training_step(image_a, image_b)


if __name__ == "__main__":
    model = CycleGanModel()