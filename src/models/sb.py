import math
import itertools
from typing import Any, Optional
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from ..ops import (
    ResnetGenerator_ncsn,
    NLayerDiscriminator_ncsn,
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
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



class SbModel(LightningModule):
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
        lambda_sb: float = 0.1,
        batch_size: int = 1,
        num_timesteps: int = 5,
        tau: float = 0.01,
        std: float = 0.25,
        ngf: int = 64,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lambda_a = 10.0
        # ResnetGenerator_ncsn
        self.g = ResnetGenerator_ncsn(
            input_nc=3,
            output_nc=3,
            ngf=64,
            norm_layer=nn.InstanceNorm2d,
            use_dropout=True,
            n_blocks=9,
            padding_type="reflect",
            no_antialias=False,
            no_antialias_up=False,
            n_mlp=3
        )
        self.f = PatchSampleF(
            use_mlp=True,
            init_type="normal",
            nc=256,
        )
        # NLayerDiscriminator_ncsn
        self.d = NLayerDiscriminator_ncsn(
            input_nc=3,
            ndf=64,
            n_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=False,
        )
        self.e = NLayerDiscriminator_ncsn(
            input_nc=3*4,
            ndf=64,
            n_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=False,
        )
        # print(self.d)
        # print(self.e)


        self.g.apply(initialize_weights)
        # self.f.apply(initialize_weights)
        self.d.apply(initialize_weights)
        self.e.apply(initialize_weights)

        self.fake_pool_a = ImagePool(50)
        self.fake_pool_b = ImagePool(50)

        self.criterion1 = GanLoss("lsgan")
        self.nce_layers = [0, 4, 8, 12, 16]
        self.criterion_nce = []
        for _ in self.nce_layers:
            self.criterion_nce.append(PatchNceLoss(batch_size))

        # fid = FrechetInceptionDistance(feature=64)
        # kid = KernelInceptionDistance(feature=64)

        image_shape = (3, 256, 256)
        # self.example_input_array = (
        #     torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        # )
        self.training_step_outputs = []
        self.training_step_outputs2 = []

    # def forward(self, x, z):
    #     image_a, _ = x, z
    #     return self.g(image_a), self.d(image_a)

    def configure_optimizers(self) -> Any:
        optimizer_g = self.hparams.optimizer1(self.g.parameters())
        optimizer_d = self.hparams.optimizer2(self.d.parameters())
        optimizer_e = self.hparams.optimizer2(self.e.parameters())

        # if self.hparams.scheduler1 is not None and self.hparams.scheduler2 is not None:
        #     scheduler_g = self.hparams.scheduler1(optimizer_g)
        #     scheduler_d = self.hparams.scheduler2(optimizer_d)
        #     return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]

        return [optimizer_g, optimizer_d, optimizer_e], []

    def calculate_nce_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z    = torch.randn(size=[src.size(0), 4 * self.hparams.ngf]).to(self.device)
        feat_q = self.g(tgt, self.time_idx*0, z, self.nce_layers, encode_only=True)

        # if self.opt.flip_equivariance and self.flipped_for_equivariance:
        #     feat_q = [torch.flip(fq, [3]) for fq in feat_q]
        
        feat_k = self.g(src, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.f(feat_k, self.hparams.num_patches, None)
        feat_q_pool, _ = self.f(feat_q, self.hparams.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterion_nce, self.nce_layers):
            loss = crit(f_q, f_k) * self.hparams.lambda_nce
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers


    def generator_training_step(self, image_a, image_b):
        bs = image_a.size(0)
        
        self.fake = self.fake_b
        std = torch.rand(size=[1]).item() * self.hparams.std

        pred_fake = self.d(self.fake, self.time_idx) #!
        loss_g_gan = self.criterion1(pred_fake, True).mean() * self.hparams.lambda_gan

        xtxt1 = torch.cat([self.real_a_noisy, self.fake_b], dim=1)
        xtxt2 = torch.cat([self.real_a_noisy2, self.fake_b2], dim=1)

        et_xy = self.e(xtxt1, self.time_idx, xtxt1).mean() \
            - torch.logsumexp(self.e(xtxt1, self.time_idx, xtxt2).reshape(-1), dim=0)
        loss_SB = -(self.hparams.num_timesteps - self.time_idx[0]) / self.hparams.num_timesteps * self.hparams.tau * et_xy
        loss_SB += self.hparams.tau * torch.mean((self.real_a_noisy - self.fake_b)**2)
        
        loss_NCE = self.calculate_nce_loss(image_a, self.fake)

        loss_NCE_Y = self.calculate_nce_loss(image_b, self.idt_b)
        loss_NCE_both = (loss_NCE + loss_NCE_Y) * 0.5

        loss_G = loss_g_gan + loss_SB + loss_NCE_both
        self.log("gen_loss", loss_G, prog_bar=True)
        return loss_G

    def discriminator_training_step(self, image_a, image_b):
        # fake_b = self.fake_b.detach()
        bs = image_a.size(0)

        self.fake = self.fake_b.detach()
        std = torch.rand(size=[1]).item() * self.hparams.std

        pred_fake = self.d(self.fake, self.time_idx)
        loss_d_fake = self.criterion1(pred_fake, False).mean()

        pred_real = self.d(image_b, self.time_idx)
        loss_d_real = self.criterion1(pred_real, True).mean()

        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.log("dis_loss", loss_d, prog_bar=True)
        return loss_d

    def e_training_step(self, image_a, image_b):
        bs = image_a.size(0)
        XtXt_1 = torch.cat([self.real_a_noisy,self.fake_b.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_a_noisy2,self.fake_b2.detach()], dim=1)
        temp = torch.logsumexp(self.e(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0).mean()
        loss_E = -self.e(XtXt_1, self.time_idx, XtXt_1).mean() +temp + temp**2
        self.log("e_loss", loss_E, prog_bar=True)
        return loss_E
        # loss_d_fake = self.criterion1(pred_fake_b, False).mean()

        # pred_real = self.d(image_b)
        # loss_d_real = self.criterion1(pred_real, True).mean()

        # loss_d = (loss_d_fake + loss_d_real) * 0.5
        # self.log("dis_loss", loss_d, prog_bar=True)
        # return loss_d

    def training_step(self, batch, batch_idx):
        image_a, image_b = batch


        tau = self.hparams.tau
        T = self.hparams.num_timesteps
        incs = np.array([0] + [1/(i+1) for i in range(T-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times, device=self.device).float()
        self.times = times
        bs = image_a.size(0)
        time_idx = (torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()).long()
        # print(time_idx)
        self.time_idx = time_idx
        self.timestep = times[time_idx]

        with torch.no_grad():
            self.g.eval()
            for t in range(self.time_idx.int().item()+1):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)

                Xt = image_a if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() \
                    + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.device)
                time_idx = (t * torch.ones(size=[image_a.shape[0]]).to(self.device)).long()
                time = times[time_idx]
                z = torch.randn(size=[image_a.shape[0],4*self.hparams.ngf]).to(self.device)
                Xt_1 = self.g(Xt, time_idx, z)
                
                Xt2 = image_a if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.device)
                time_idx = (t * torch.ones(size=[image_a.shape[0]]).to(self.device)).long()
                time = times[time_idx]
                z = torch.randn(size=[image_a.shape[0],4*self.hparams.ngf]).to(self.device)
                Xt_12 = self.g(Xt2, time_idx, z)
                
                # if self.opt.nce_idt:
                XtB = image_b if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.device)
                time_idx = (t * torch.ones(size=[image_a.shape[0]]).to(self.device)).long()
                time = times[time_idx]
                z = torch.randn(size=[image_a.shape[0],4*self.hparams.ngf]).to(self.device)
                Xt_1B = self.g(XtB, time_idx, z)

            # if self.opt.nce_idt:
            self.XtB = XtB.detach()

            self.real_a_noisy = Xt.detach()
            self.real_a_noisy2 = Xt2.detach()
                      
        
        z_in = torch.randn(size=[2*bs,4*self.hparams.ngf]).to(image_a.device)
        z_in2 = torch.randn(size=[bs,4*self.hparams.ngf]).to(image_a.device)
        """Run forward pass"""
        self.real = torch.cat((image_a, image_b), dim=0) if self.hparams.nce_idt and self.training else image_a
        
        self.realt = torch.cat((self.real_a_noisy, self.XtB), dim=0) if self.hparams.nce_idt and self.training else self.real_a_noisy
        
        # if self.opt.flip_equivariance:
        #     self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
        #     if self.flipped_for_equivariance:
        #         self.real = torch.flip(self.real, [3])
        #         self.realt = torch.flip(self.realt, [3])
        
        self.fake = self.g(self.realt, self.time_idx, z_in)
        self.fake_b2 = self.g(self.real_a_noisy2, self.time_idx, z_in2)
        self.fake_b = self.fake[:image_a.size(0)]
        if self.hparams.nce_idt:
            self.idt_b = self.fake[image_a.size(0):]
            
        optimizer_g, optimizer_d, optimizer_e = self.optimizers()
        # scheduler_g, scheduler_d = self.lr_schedulers()

        # if self.hparams.lambda_nce > 0:
        #     optimizer_f = self.hparams.optimizer2(self.f.parameters())
        # print(self.d)
        # for p in self.d.parameters():
        #     print(p.requires_grad)
        set_requires_grad(self.d, True)
        optimizer_d.zero_grad()
        d_loss = self.discriminator_training_step(image_a, image_b)
        self.manual_backward(d_loss)
        optimizer_d.step()
        # scheduler_d.step()
        # self.untoggle_optimizer(optimizer_d)

        set_requires_grad(self.e, True)
        optimizer_e.zero_grad()
        e_loss = self.e_training_step(image_a, image_b)
        self.manual_backward(e_loss)
        optimizer_e.step()

        set_requires_grad(self.d, False)
        set_requires_grad(self.e, False)

        optimizer_g.zero_grad()
        # optimizer_f.zero_grad()
        g_loss = self.generator_training_step(image_a, image_b)
        self.manual_backward(g_loss)
        optimizer_g.step()

        # optimizer_f.step()
        # scheduler_g.step()
        # self.untoggle_optimizer(optimizer_g)
        if len(self.training_step_outputs) < 7:
            fake_b = self.g(image_a, time_idx, z).detach()

            # fake_a = self.g.y(image_b).detach()
            # reco_b = self.g.x(fake_a).detach()
            self.training_step_outputs.append(image_a)
            self.training_step_outputs2.append(fake_b)


if __name__ == "__main__":
    model = SbModel()
