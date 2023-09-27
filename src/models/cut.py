import itertools
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
    Initializer
)


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
        optimizer1: optim.Optimizer,
        optimizer2: optim.Optimizer,
        scheduler1: Optional[Any] = None,
        scheduler2: Optional[Any] = None,
        image_size: int = 256,
        lambda_a: float = 10.0,
        lambda_b: float = 10.0,
        lambda_i: float = 0.5,
        lambda_nce: float = 1.0,  # fastcut 10.0
        lambda_gan: float = 1.0,
        nce_idt: bool = True,  # fastcut False
        pool_size: int = 0,
        num_patches: int = 256,
        batch_size: int = 1,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lambda_a = 10.0
        self.g = ResnetGenerator(
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
        self.f = PatchSampleF(
            use_mlp=True,
            init_type="normal",
            nc=256,
        )
        self.d = NLayerDiscriminator(
            input_nc=3,
            ndf=64,
            num_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=True,
        )


        self.g.apply(initialize_weights)
        # self.f.apply(initialize_weights)
        self.d.apply(initialize_weights)

        # self.fake_pool_a = ImagePool(0)
        # self.fake_pool_b = ImagePool(0)

        self.criterion1 = GanLoss("lsgan")
        self.nce_layers = [0, 4, 8, 12, 16]
        self.criterion_nce = []
        for _ in self.nce_layers:
            self.criterion_nce.append(PatchNceLoss(batch_size))

        # fid = FrechetInceptionDistance(feature=64)
        # kid = KernelInceptionDistance(feature=64)

        image_shape = (3, 256, 256)
        self.example_input_array = (
            torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        )
        self.training_step_outputs = []
        self.training_step_outputs2 = []

    def forward(self, x, z):
        image_a, _ = x, z
        return self.g(image_a), self.d(image_a)

    def configure_optimizers(self) -> Any:
        optimizer_g = self.hparams.optimizer1(self.g.parameters())
        optimizer_d = self.hparams.optimizer2(self.d.parameters())

        if self.hparams.scheduler1 is not None and self.hparams.scheduler2 is not None:
            scheduler_g = self.hparams.scheduler1(optimizer_g)
            scheduler_d = self.hparams.scheduler2(optimizer_d)
            return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

        return [optimizer_g, optimizer_d], []

    def calculate_nce_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        # print(src.shape, tgt.shape)
        feat_q = self.g(tgt, self.nce_layers, encode_only=True)

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.g(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.f(feat_k, self.hparams.num_patches, None)
        feat_q_pool, _ = self.f(feat_q, self.hparams.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, _ in zip(feat_q_pool, feat_k_pool, self.criterion_nce, self.nce_layers):
            loss = crit(f_q, f_k) * self.hparams.lambda_nce
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def generator_training_step(self, image_a, image_b):
        """Calculate GAN loss for the generator"""
        # if self.hparams.nce_idt and self.training:
        #     reals = torch.cat((image_a, image_b), dim=0)
        #     # TODO: if flipped_for_equivariance:  ## fastcut
        
        # # fakes = self.g(reals)
        # self.fake_b = self.fakes[:image_a.size(0)]

        if self.hparams.nce_idt:
            idt_b = self.fakes[image_a.size(0):]

        # [B, 1, 30, 30]
        pred_fake_b = self.d(self.fake_b)
        loss_g_gan = self.criterion1(pred_fake_b, True).mean() * self.hparams.lambda_gan
        # print(image_a.shape, pred_fake_b.shape, idt_b.shape, image_b.shape)

        loss_nce = self.calculate_nce_loss(image_a, self.fake_b)
        loss_nce_y = self.calculate_nce_loss(image_b, idt_b)
        loss_nce = (loss_nce + loss_nce_y) * 0.5
        gen_loss = loss_g_gan + loss_nce
        self.log("gen_loss", gen_loss, prog_bar=True)
        return gen_loss

    def discriminator_training_step(self, image_a, image_b):
        fake_b = self.fake_b.detach()

        pred_fake_b = self.d(fake_b)
        loss_d_fake = self.criterion1(pred_fake_b, False).mean()

        pred_real = self.d(image_b)
        loss_d_real = self.criterion1(pred_real, True).mean()

        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.log("dis_loss", loss_d, prog_bar=True)
        return loss_d

    def training_step(self, batch, batch_idx):
        image_a, image_b = batch

        if self.hparams.nce_idt:
            reals = torch.cat((image_a, image_b), dim=0)
            # TODO: if flipped_for_equivariance:  ## fastcut
        
        self.fakes = self.g(reals)
        self.fake_b = self.fakes[:image_a.size(0)]

        optimizer_g, optimizer_d = self.optimizers()
        # scheduler_g, scheduler_d = self.lr_schedulers()

        # if self.hparams.lambda_nce > 0:
        #     optimizer_f = self.hparams.optimizer2(self.f.parameters())

        self.toggle_optimizer(optimizer_d)
        optimizer_d.zero_grad()
        d_loss = self.discriminator_training_step(image_a, image_b)
        self.manual_backward(d_loss)
        optimizer_d.step()
        # scheduler_d.step()
        self.untoggle_optimizer(optimizer_d)

        self.toggle_optimizer(optimizer_g)
        optimizer_g.zero_grad()
        # optimizer_f.zero_grad()
        g_loss = self.generator_training_step(image_a, image_b)
        self.manual_backward(g_loss)
        optimizer_g.step()
        # optimizer_f.step()
        # scheduler_g.step()
        self.untoggle_optimizer(optimizer_g)



        if len(self.training_step_outputs) < 7:
            fake_b = self.g(image_a).detach()

            # fake_a = self.g.y(image_b).detach()
            # reco_b = self.g.x(fake_a).detach()
            self.training_step_outputs.append(image_a)
            self.training_step_outputs2.append(fake_b)


if __name__ == "__main__":
    model = CutModel()