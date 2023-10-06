import itertools
from typing import Any, Optional
from enum import IntEnum
from lightning.pytorch.utilities.types import STEP_OUTPUT

import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from ..ops import ResnetGenerator, NLayerDiscriminator, ImagePool, initialize_weights, Initializer, GanLoss, set_requires_grad
from ..datasets import inversed_transform


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        expanded_channels: int = 64,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        use_dropout: bool = False,
        num_blocks: int = 9,
        padding_layer: nn.Module = nn.ReflectionPad2d,
        no_antialias: bool = False,
        no_antialias_up: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.x = ResnetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            expanded_channels=expanded_channels,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            num_blocks=num_blocks,
            padding_layer=padding_layer,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias,
            use_bias=use_bias
        )
        self.y = ResnetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            expanded_channels=expanded_channels,
            norm_layer=norm_layer,
            use_dropout=use_dropout,
            num_blocks=num_blocks,
            padding_layer=padding_layer,
            no_antialias=no_antialias,
            no_antialias_up=no_antialias_up,
            use_bias=use_bias
        )


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        expanded_channels: int = 64,
        num_layers: int = 3,
        norm_layer: nn.Module = nn.InstanceNorm2d,
        no_antialias: bool = False,
    ) -> None:
        super().__init__()
        self.x = NLayerDiscriminator(
            in_channels=in_channels,
            expanded_channels=expanded_channels,
            num_layers=num_layers,
            norm_layer=norm_layer,
            no_antialias=no_antialias,
        )
        self.y = NLayerDiscriminator(
            in_channels=in_channels,
            expanded_channels=expanded_channels,
            num_layers=num_layers,
            norm_layer=norm_layer,
            no_antialias=no_antialias,
        )


class CycleGanModel(LightningModule):
    """
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf

    Args:
        input_channels: default=3
        expanded_channels: default=64
        lambda_a: default=10.0 weight for cycle loss (A -> B -> A) forward cycle loss
        lambda_b: default=10.0 weight for cycle loss (B -> A -> B) backward cycle loss
        lambda_i: default=0.5 use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. 
            For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, 
            please set lambda_identity = 0.1. lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
            Dropout is not used in the original CycleGAN paper.
        pool_size: default=50  the size of image buffer that stores previously generated images
        gan_mode: default=lsgan | vanilla, lsgan | wgangp
        lr: default=0.0002  initial learning rate for adam
        beta1: default=0.5  momentum term of adam
        beta2: default=0.999  momentum term of adam  
        lr_policy: default=linear | linear, step, plateau, cosine
        lr_decay_iters: default=50  multiply by a gamma every lr_decay_iters iterations
    """
    def __init__(
        self,
        generator,
        discriminator,
        optimizer1: optim.Optimizer,
        optimizer2: optim.Optimizer,
        scheduler1: Optional[Any] = None,
        scheduler2: Optional[Any] = None,
        init_type: Initializer = Initializer.XAVIER_NORMAL,
        init_gain: float = 0.02,
        lambda_a: float = 10.0,
        lambda_b: float = 10.0,
        lambda_i: float = 0.5,
        pool_size: int = 50,
        image_shape: tuple = (3, 256, 256),
        **kwargs
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator"])
        self.automatic_optimization = False

        self.g = generator
        self.d = discriminator
        initialize_weights(self.g.x, init_type=init_type, gain=init_gain)
        initialize_weights(self.g.y, init_type=init_type, gain=init_gain)
        initialize_weights(self.d.x, init_type=init_type, gain=init_gain)
        initialize_weights(self.d.y, init_type=init_type, gain=init_gain)

        self.image_pool_a = ImagePool(pool_size)
        self.image_pool_b = ImagePool(pool_size)

        self.example_input_array = (
            torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        )
        self.training_step_outputs = []
        self.training_step_outputs2 = []

        self.criterion_gan = GanLoss("lsgan")
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        self.metric_fid = FrechetInceptionDistance(feature=2048)
        # self.metric_kid = KernelInceptionDistance(feature=64)

    def forward(self, x, z):
        return self.g.x(x), self.g.y(z), self.d.x(x), self.d.y(z)

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
            return [
                {
                    "optimizer": optimizer_g,
                    "lr_scheduler":
                        {"scheduler": scheduler_g, "interval": "epoch"}
                },
                {
                    "optimizer": optimizer_d,
                    "lr_scheduler": 
                        {"scheduler": scheduler_d, "interval": "epoch"}
                }
            ]

        return [optimizer_g, optimizer_d], []

    def generator_training_step(self, image_a, image_b, fake_a, fake_b, cycle_a, cycle_b):
        """Calculate GAN loss for the generator"""
        if self.hparams.lambda_i > 0:
            same_a = self.g.x(image_b)
            loss_i_a = \
                self.criterion_identity(same_a, image_b) * self.hparams.lambda_b * self.hparams.lambda_i
            self.log("identity_a", loss_i_a, prog_bar=True)
    
            same_b = self.g.y(image_a)
            loss_i_b = \
                self.criterion_identity(same_b, image_a) * self.hparams.lambda_a * self.hparams.lambda_i
            self.log("identity_b", loss_i_b, prog_bar=True)
    
        # GAN loss D_A(G_A(A))
        loss_d_x = self.criterion_gan(self.d.x(fake_b), real=True)
        
        # GAN loss D_B(G_B(B))
        loss_d_y = self.criterion_gan(self.d.y(fake_a), real=True)
        # Identity loss
        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        loss_cycle_a = self.criterion_cycle(cycle_a, image_a) * self.hparams.lambda_a
        self.log("cycle_a", loss_cycle_a, prog_bar=True)

        loss_cycle_b = self.criterion_cycle(cycle_b, image_b) * self.hparams.lambda_b
        self.log("cycle_b", loss_cycle_b, prog_bar=True)

        g_loss = loss_d_x + loss_d_y + loss_cycle_a + loss_cycle_b + loss_i_a + loss_i_b
        self.log("total_g_loss", g_loss, prog_bar=True)

        return g_loss

    def discriminator_training_step(self, image_a, image_b, fake_a, fake_b) -> None:
        """Calculate GAN loss for the discriminator
        """
        # Calculate GAN loss for discriminator D_A, D_B
        pred_real_a = self.d.x(image_b)
        loss_dx_real = self.criterion_gan(pred_real_a, True)
        fake_b = self.image_pool_b.query(fake_b)
        pred_fake_a = self.d.x(fake_b.detach())
        loss_dx_fake = self.criterion_gan(pred_fake_a, False)
        dx_loss = (loss_dx_real + loss_dx_fake) * 0.5
        self.log("dx_loss", dx_loss, prog_bar=True)

        pred_real_b = self.d.y(image_a)
        loss_dy_real = self.criterion_gan(pred_real_b, True)
        fake_a = self.image_pool_a.query(fake_a)
        pred_fake_b = self.d.y(fake_a.detach())
        loss_dy_fake = self.criterion_gan(pred_fake_b, False)
        dy_loss = (loss_dy_real + loss_dy_fake) * 0.5

        self.log("dy_loss", dy_loss, prog_bar=True)
        return dx_loss, dy_loss

    def training_step(self, batch, batch_idx):
        image_a, image_b = batch
        fake_b = self.g.x(image_a)
        cycle_a = self.g.y(fake_b)
        fake_a = self.g.y(image_b)
        cycle_b = self.g.x(fake_a)

        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        g_loss = self.generator_training_step(image_a, image_b, fake_a, fake_b, cycle_a, cycle_b)
        self.manual_backward(g_loss)
        optimizer_g.step()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        dx_loss, dy_loss = self.discriminator_training_step(image_a, image_b, fake_a, fake_b)
        self.manual_backward(dx_loss)
        self.manual_backward(dy_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)
        
        if len(self.training_step_outputs) < 7:
            with torch.no_grad():
                fake_b = self.g.x(image_a).detach()
                self.training_step_outputs.append(image_a)
                self.training_step_outputs2.append(fake_b)

    def on_train_epoch_end(self) -> None:
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()
        # If the selected scheduler is a ReduceLROnPlateau scheduler.
        if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_g.step(self.trainer.callback_metrics["loss"])

    def validation_step(self, batch, batch_idx):
        image_a, image_b = batch
        fake_b = self.g.x(image_a)

        image_b = inversed_transform(image_b)
        fake_b = inversed_transform(fake_b)

        self.metric_fid.update(image_b, real=True)
        self.metric_fid.update(fake_b, real=False)
        
    def on_validation_epoch_end(self):
        self.metric_fid.compute()
        self.log("valid/fid", self.metric_fid, prog_bar=True)


if __name__ == "__main__":
    model = CycleGanModel()
