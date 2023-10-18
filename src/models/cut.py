import itertools
import random
from typing import Any, Optional

import torch
from torch import nn, optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from ..ops import (
    ResnetGenerator,
    NLayerDiscriminator,
    ImagePool,
    PatchSampleF,
    GanLoss,
    PatchNceLoss,
    initialize_weights,
    Initializer,
    set_requires_grad
)
from ..datasets import inversed_transform, InverseNormalize, inversed_transform_ir


class CutModel(LightningModule):
    """
    nce_layers: list of layers to apply nce loss
    nce_idt (bool): whether to apply nce loss to identity mapping
    lambda_nce (float): weight for nce loss NCE(G(X), X)
    lambda_gan (float): weight for gan loss GAN(G(X))
    nce_t (float): temperature for nce loss
    pool_size (int): size of image buffer that stores previously generated images
    """
    def __init__(
        self,
        optimizer1: optim.Optimizer = None,
        optimizer2: optim.Optimizer = None,
        optimizer3: optim.Optimizer = None,
        scheduler1: Optional[Any] = None,
        scheduler2: Optional[Any] = None,
        scheduler3: Optional[Any] = None,
        image_size: int = 256,
        lambda_nce: float = 1.0,  # fastcut 10.0
        nce_layers: list[int] = [0, 4, 8, 12, 16],
        lambda_gan: float = 1.0,
        nce_idt: bool = True,  # fastcut False
        nce_t: float = 0.07,
        num_patches: int = 256,
        flip_equivariance: bool = False,
        batch_size: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = False,
        image_shape: tuple[int, int, int] = (3, 256, 256),
        in_channels: int = 3,
        out_channels: int = 3,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.g = ResnetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            expanded_channels=64,
            norm_layer=nn.InstanceNorm2d,
            use_dropout=True,
            num_blocks=9,
            padding_layer=nn.ReflectionPad2d,
            no_antialias=False,
            no_antialias_up=False
        )
        self.f = PatchSampleF(
            use_mlp=True,
            init_type=1,
            nc=256,
        )
        self.d = NLayerDiscriminator(
            in_channels=in_channels, # 3 channels
            expanded_channels=64,
            num_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=False,
        )

        initialize_weights(self.g, init_type=1, gain=0.02)
        initialize_weights(self.d, init_type=1, gain=0.02)

        self.criterion1 = GanLoss("lsgan")
        self.criterion_nce = []
        for _ in self.hparams.nce_layers:
            self.criterion_nce.append(PatchNceLoss(batch_size))

        self.metric_fid = FrechetInceptionDistance(feature=2048)

        self.example_input_array = (
            torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        )
        self.training_step_outputs1 = []
        self.training_step_outputs2 = []
        self.validation_step_outputs1 = []
        self.validation_step_outputs2 = []

    def forward(self, x, z):
        return self.g(x)

    def configure_optimizers(self) -> Any:
        return_list = []
        optimizer_g = self.hparams.optimizer1(self.g.parameters())
        optimizer_d = self.hparams.optimizer2(self.d.parameters())
        return_list.append({"optimizer": optimizer_g})
        return_list.append({"optimizer": optimizer_d})

        if self.hparams.scheduler1 is not None and self.hparams.scheduler2 is not None:
            scheduler_g = self.hparams.scheduler1(optimizer_g)
            scheduler_d = self.hparams.scheduler2(optimizer_d)
            return_list[0].update({"lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch"}})
            return_list[1].update({"lr_scheduler": {"scheduler": scheduler_d, "interval": "epoch"}})
    
        return return_list

    def compute_nce_loss(self, source, target):
        feat_q = self.g(target, self.hparams.nce_layers, encode_only=True)

        if self.hparams.flip_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.g(source, self.hparams.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.f(feat_k, self.hparams.num_patches, None)
        feat_q_pool, _ = self.f(feat_q, self.hparams.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, criterion, _ in zip(feat_q_pool, feat_k_pool, self.criterion_nce, self.hparams.nce_layers):
            loss = criterion(f_q, f_k) * self.hparams.lambda_nce
            total_nce_loss += loss.mean()

        return total_nce_loss / len(self.hparams.nce_layers)

    def generator_training_step(self, image_a, image_b, fake_b, idt_b=None):
        """Calculate GAN loss for the generator"""
        # [B, 1, 30, 30]
        # if self.hparams.lambda_gan > 0:            
        pred_fake_b = self.d(fake_b)
        loss_g_gan = self.criterion1(pred_fake_b, True).mean() * self.hparams.lambda_gan

        # if self.hparams.lambda_nce > 0:
        loss_nce = self.compute_nce_loss(image_a, fake_b)
        
        loss_nce_y = 0.0
        if self.hparams.nce_idt:
            loss_nce_y = self.compute_nce_loss(image_b, idt_b)
    
        loss_nce = (loss_nce + loss_nce_y) * 0.5
        self.log("loss_nce", loss_nce, prog_bar=True)
        g_loss = loss_g_gan + loss_nce
        self.log("loss_g", g_loss, prog_bar=True)
        return g_loss

    def discriminator_training_step(self, image_a, image_b, fake_b):
        pred_fake_b = self.d(fake_b.detach())
        loss_d_fake = self.criterion1(pred_fake_b, False).mean()

        pred_real = self.d(image_b)
        loss_d_real = self.criterion1(pred_real, True).mean()

        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.log("loss_d", loss_d, prog_bar=True)
        return loss_d
    
    def on_fit_start(self) -> None:
        # if self.device: "cuda:0"
        sample = self.example_input_array[0].cuda()
        sample = self.g(sample, self.hparams.nce_layers, encode_only=True)
        _ = self.f(sample, self.hparams.num_patches, None)
        if self.hparams.lambda_nce > 0:
            self.optimizer_f = self.hparams.optimizer3(self.f.parameters())
            self.scheduler_f = self.hparams.scheduler3(self.optimizer_f)

    def training_step(self, batch, batch_idx):
        image_a, image_b = batch

        if self.hparams.nce_idt:
            reals = torch.cat((image_a, image_b), dim=0)
        else:
            reals = image_a

        if self.hparams.flip_equivariance:
            if random.random() < 0.5:
                reals = torch.flip(reals, [3])

        fakes = self.g(reals)
        fake_b = fakes[:image_a.size(0)]

        idt_b = None
        if self.hparams.nce_idt:
            idt_b = fakes[image_a.size(0):]

        optimizer_g, optimizer_d = self.optimizers()

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        d_loss = self.discriminator_training_step(image_a, image_b, fake_b)
        self.manual_backward(d_loss)
        optimizer_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_g)
        # self.toggle_optimizer(self.optimizer_f)
        set_requires_grad(self.f, True)
        optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        g_loss = self.generator_training_step(image_a, image_b, fake_b, idt_b)
        self.manual_backward(g_loss)
        optimizer_g.step()
        self.optimizer_f.step()
        self.untoggle_optimizer(optimizer_g)
        set_requires_grad(self.f, False)
        # self.untoggle_optimizer(self.optimizer_f)

        # if torch.cat(self.training_step_outputs1, dim=0).size(0) < 10:
        if len(self.training_step_outputs1) < 10:
            with torch.no_grad():
                fake_b = self.g(image_a)
                self.training_step_outputs1.append(image_a.detach())
                self.training_step_outputs2.append(fake_b.detach())

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()
        self.scheduler_f.step()

    def validation_step(self, batch, batch_idx):
        image_a, image_b = batch
        fake_b = self.g(image_a)

        image_b = inversed_transform(image_b)
        fake_b = inversed_transform(fake_b)

        # rgb_transform = InverseNormalize()
        # ir_transform = InverseNormalize(mean=[-1.], std=[2.])

        # ! delete for original
        # image_b = inversed_transform_ir(image_b)
        # fake_b = inversed_transform_ir(fake_b)

        # ! grayscale
        # image_b = torch.cat([image_b, image_b, image_b], dim=1)
        # fake_b = torch.cat([fake_b, fake_b, fake_b], dim=1)

        self.metric_fid.update(image_b, real=True)
        self.metric_fid.update(fake_b, real=False)

        # if len(self.validation_step_outputs1) < 10:
        #     self.validation_step_outputs1.append(image_a.detach())
        #     self.validation_step_outputs2.append(fake_b.detach())

    def on_validation_epoch_end(self):
        self.metric_fid.compute()
        self.log("valid/fid", self.metric_fid, prog_bar=True)


if __name__ == "__main__":
    model = CutModel()