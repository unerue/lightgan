import math
import itertools
from typing import Any, Optional
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
# from torchmetrics.image.kid import KernelInceptionDistance
from ..datasets import inversed_transform

from ..ops import (
    NcsnResnetGenerator,
    NcsnNLayerDiscriminator,
    PatchSampleF,
    GanLoss,
    PatchNceLoss,
    initialize_weights,
    Initializer,
    set_requires_grad
)


class SbModel(LightningModule):
    """
    Args:
        nce_layers: list of layers to apply nce loss
        nce_idt (bool): whether to apply nce loss to identity mapping
        lambda_nce (float): weight for nce loss NCE(G(X), X)
        lambda_gan (float): weight for gan loss GAN(G(X))
        nce_t (float): temperature for nce loss
    """
    def __init__(
        self,
        optimizer1: optim.Optimizer,
        optimizer2: optim.Optimizer,
        optimizer3: optim.Optimizer,
        scheduler1: Optional[Any] = None,
        scheduler2: Optional[Any] = None,
        scheduler3: Optional[Any] = None,
        image_size: int = 256,
        expended_channels: int = 64,
        lambda_sb: float = 1.0,
        lambda_nce: float = 1.0,
        lambda_gan: float = 1.0,
        nce_t: float = 0.07,
        nce_idt: bool = True,
        nce_layers: list[int] = [0, 4, 8, 12, 16],
        pool_size: int = 0,
        num_patches: int = 256,
        batch_size: int = 1,
        num_timesteps: int = 5,
        tau: float = 0.01,
        std: float = 0.25,
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lambda_a = 10.0
        self.g = NcsnResnetGenerator(
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
        self.d = NcsnNLayerDiscriminator(
            input_nc=3,
            ndf=64,
            n_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=False,
        )
        self.e = NcsnNLayerDiscriminator(
            input_nc=3*4,
            ndf=64,
            n_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=False,
        )

        initialize_weights(self.g, init_type=1, gain=0.02)
        initialize_weights(self.d, init_type=1, gain=0.02)
        initialize_weights(self.e, init_type=1, gain=0.02)

        self.criterion1 = GanLoss("lsgan")
        self.criterion_nce = []
        for _ in self.hparams.nce_layers:
            self.criterion_nce.append(PatchNceLoss(self.hparams.batch_size))

        self.metric_fid = FrechetInceptionDistance(feature=2048)

        image_shape = (3, 256, 256)
        self.example_input_array = (
            torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        )
        self.training_step_outputs1 = []
        self.training_step_outputs2 = []

    def forward(self, x, z):
        image_a, _ = x, z
        z = torch.randn(size=[1, 4*64], device=self.device)
        time_idx = torch.randint(5, size=[1], dtype=torch.int64, device=self.device) \
            * torch.ones(size=[1], dtype=torch.int64, device=self.device)
        return self.g(image_a, time_idx, z)

    def configure_optimizers(self) -> list[dict]:
        return_list = []
        optimizer_g = self.hparams.optimizer1(self.g.parameters())
        optimizer_d = self.hparams.optimizer2(self.d.parameters())
        optimizer_e = self.hparams.optimizer3(self.e.parameters())
        return_list.append({"optimizer": optimizer_g})
        return_list.append({"optimizer": optimizer_d})
        return_list.append({"optimizer": optimizer_e})

        if self.hparams.scheduler1 is not None and self.hparams.scheduler2 is not None:
            scheduler_g = self.hparams.scheduler1(optimizer_g)
            scheduler_d = self.hparams.scheduler2(optimizer_d)
            scheduler_e = self.hparams.scheduler3(optimizer_e)
            return_list[0].update({"lr_scheduler": {"scheduler": scheduler_g, "interval": "epoch"}})
            return_list[1].update({"lr_scheduler": {"scheduler": scheduler_d, "interval": "epoch"}})
            return_list[2].update({"lr_scheduler": {"scheduler": scheduler_e, "interval": "epoch"}})
    
        return return_list

    def compute_nce_loss(self, src, tgt, time_idx):
        z = torch.randn([src.size(0), 4 * self.hparams.expended_channels], device=self.device)
        feat_q = self.g(tgt, time_idx*0, z, self.hparams.nce_layers, encode_only=True)
        
        feat_k = self.g(src, time_idx * 0, z, self.hparams.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.f(feat_k, self.hparams.num_patches, None)
        feat_q_pool, _ = self.f(feat_q, self.hparams.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, criterion, _ in zip(feat_q_pool, feat_k_pool, self.criterion_nce, self.hparams.nce_layers):
            loss = criterion(f_q, f_k) * self.hparams.lambda_nce
            total_nce_loss += loss.mean()

        return total_nce_loss / len(self.hparams.nce_layers)

    def generator_training_step(self, image_a, image_b, fake_b1, fake_b2, real_noisy1, real_noisy2, idt_b, time_idx):
        pred_fake = self.d(fake_b1, time_idx)
        loss_g_gan = self.criterion1(pred_fake, True).mean() * self.hparams.lambda_gan

        xtxt1 = torch.cat([real_noisy1, fake_b1], dim=1)
        xtxt2 = torch.cat([real_noisy2, fake_b2], dim=1)

        et_xy = self.e(xtxt1, time_idx, xtxt1).mean() \
            - torch.logsumexp(self.e(xtxt1, time_idx, xtxt2).reshape(-1), dim=0)
        loss_sb = -(self.hparams.num_timesteps - time_idx[0]) \
            / self.hparams.num_timesteps * self.hparams.tau * et_xy
        loss_sb += self.hparams.tau * torch.mean((real_noisy1 - fake_b1)**2)

        loss_nce = self.compute_nce_loss(image_a, fake_b1, time_idx)
        loss_nce_y = self.compute_nce_loss(image_b, idt_b, time_idx)
        loss_nce_both = (loss_nce + loss_nce_y) * 0.5
        self.log("loss_nce", loss_nce_both, prog_bar=True)

        loss_g = loss_g_gan + loss_sb + loss_nce_both
        self.log("loss_g", loss_g, prog_bar=True)
        return loss_g

    def discriminator_training_step(self, image_a, image_b, fake_b1, time_idx):
        # std = torch.rand(size=[1]).item() * self.hparams.std

        pred_fake = self.d(fake_b1.detach(), time_idx)
        loss_d_fake = self.criterion1(pred_fake, False).mean()

        pred_real = self.d(image_b, time_idx)
        loss_d_real = self.criterion1(pred_real, True).mean()

        loss_d = (loss_d_fake + loss_d_real) * 0.5
        self.log("loss_d", loss_d, prog_bar=True)
        return loss_d

    def e_training_step(self, real_noisy1, real_noisy2, fake_b1, fake_b2, time_idx):
        xtxt1 = torch.cat([real_noisy1, fake_b1.detach()], dim=1)
        xtxt2 = torch.cat([real_noisy2, fake_b2.detach()], dim=1)
        temp = torch.logsumexp(self.e(xtxt1, time_idx, xtxt2).reshape(-1), dim=0).mean()
        loss_e = -self.e(xtxt1, time_idx, xtxt1).mean() + temp + temp**2
        loss_e.requires_grad_(True)
        self.log("loss_e", loss_e, prog_bar=True)
        return loss_e

    def on_fit_start(self) -> None:
        # if self.device: "cuda:0"
        z = torch.randn(size=[1, 4*64], device=self.device)
        time_idx = torch.randint(5, size=[1], dtype=torch.int64, device=self.device) \
            * torch.ones(size=[1], dtype=torch.int64, device=self.device)
        sample = self.example_input_array[0].to(self.device)
        sample = self.g(sample, time_idx * 0, z, self.hparams.nce_layers, encode_only=True)    
        _ = self.f(sample, self.hparams.num_patches, None)
        if self.hparams.lambda_nce > 0:
            self.optimizer_f = self.hparams.optimizer3(self.f.parameters())
            self.scheduler_f = self.hparams.scheduler3(self.optimizer_f)

    def preprocessing_inputs(self, image_a, image_b):
        tau = self.hparams.tau
        incs = np.array([0] + [1/(i+1) for i in range(self.hparams.num_timesteps-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times, dtype=torch.float32, device=self.device)

        bs = image_a.size(0)
        time_idx = (torch.randint(self.hparams.num_timesteps, size=[1], dtype=torch.int64, device=self.device) \
                    * torch.ones([image_a.size(0)], dtype=torch.int64, device=self.device))

        with torch.no_grad():
            # self.g.eval()
            for t in range(time_idx.int().item()+1):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)

                Xt = image_a if t == 0 else (1-inter) * Xt + inter * Xt_1.detach() \
                    + (scale * tau).sqrt() * torch.randn_like(Xt, device=self.device)
                time_idx = t * torch.ones([image_a.size(0)], dtype=torch.int64, device=self.device)
                z = torch.randn([image_a.size(0), 4*self.hparams.expended_channels], device=self.device)
                Xt_1 = self.g(Xt, time_idx, z)
                
                Xt2 = image_a if t == 0 else (1-inter) * Xt2 + inter * Xt_12.detach() \
                    + (scale * tau).sqrt() * torch.randn_like(Xt2, device=self.device)
                time_idx = t * torch.ones([image_a.size(0)], dtype=torch.int64, device=self.device)
                z = torch.randn([image_a.shape[0], 4*self.hparams.expended_channels], device=self.device)
                Xt_12 = self.g(Xt2, time_idx, z)
                
                if self.hparams.nce_idt:
                    XtB = image_b if t == 0 else (1-inter) * XtB + inter * Xt_1B.detach() \
                        + (scale * tau).sqrt() * torch.randn_like(XtB, device=self.device)
                    time_idx = (t * torch.ones(size=[image_a.shape[0]], device=self.device)).long()
                    z = torch.randn([image_a.size(0), 4*self.hparams.expended_channels], device=self.device)
                    Xt_1B = self.g(XtB, time_idx, z)

            if self.hparams.nce_idt:
                XtB = XtB.detach()

            real_noisy1 = Xt.detach()
            real_noisy2 = Xt2.detach()

        z_in1 = torch.randn(size=[2*bs, 4*self.hparams.expended_channels], device=self.device)
        z_in2 = torch.randn(size=[bs, 4*self.hparams.expended_channels], device=self.device)
        # real = torch.cat((image_a, image_b), dim=0) if self.hparams.nce_idt else image_a
        realt = torch.cat((real_noisy1, XtB), dim=0) if self.hparams.nce_idt else real_noisy1

        fake = self.g(realt, time_idx, z_in1)
        fake_b2 = self.g(real_noisy2, time_idx, z_in2)
        fake_b1 = fake[:image_a.size(0)]
        if self.hparams.nce_idt:
            idt_b = fake[image_a.size(0):]

        return fake_b1, fake_b2, real_noisy1, real_noisy2, idt_b, time_idx, z 

    def training_step(self, batch, batch_idx):
        image_a, image_b = batch

        fake_b1, fake_b2, real_noisy1, real_noisy2, idt_b, time_idx, z \
            = self.preprocessing_inputs(image_a, image_b)

        optimizer_g, optimizer_d, optimizer_e = self.optimizers()

        set_requires_grad(self.d, True)
        optimizer_d.zero_grad()
        loss_d = self.discriminator_training_step(image_a, image_b, fake_b1, time_idx)
        self.manual_backward(loss_d)
        optimizer_d.step()

        set_requires_grad(self.e, True)
        optimizer_e.zero_grad()
        loss_e = self.e_training_step(
            real_noisy1, real_noisy2, fake_b1, fake_b2, time_idx
        )
        self.manual_backward(loss_e)
        optimizer_e.step()

        set_requires_grad(self.d, False)
        set_requires_grad(self.e, False)

        optimizer_g.zero_grad()
        self.optimizer_f.zero_grad()
        loss_g = self.generator_training_step(
            image_a, image_b, fake_b1, fake_b2,
            real_noisy1, real_noisy2, idt_b, time_idx
        )
        self.manual_backward(loss_g)
        optimizer_g.step()
        self.optimizer_f.step()

        if len(self.training_step_outputs1) < 10:
            fake_b = self.g(image_a, time_idx, z).detach()
            self.training_step_outputs1.append(image_a)
            self.training_step_outputs2.append(fake_b)

    def on_train_epoch_end(self) -> None:
        for scheduler in self.lr_schedulers():
            scheduler.step()
        self.scheduler_f.step()

    def validation_step(self, batch, batch_idx):
        image_a, image_b = batch

        incs = np.array([0] + [1/(i+1) for i in range(self.hparams.num_timesteps-1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times, dtype=torch.float32, device=self.device)
        time_idx = (torch.randint(self.hparams.num_timesteps, size=[1], device=self.device) \
            * torch.ones([image_a.size(0)], device=self.device)).long()
        # fake_bs = []
        with torch.no_grad():
            for t in range(self.hparams.num_timesteps):
                if t > 0:
                    delta = times[t] - times[t-1]
                    denom = times[-1] - times[t-1]
                    inter = (delta / denom).reshape(-1,1,1,1)
                    scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)
                xt = image_a if t == 0 else (1-inter) * xt + inter * fake_b.detach() \
                    + (scale * self.hparams.tau).sqrt() * torch.randn_like(xt, device=self.device)
                time_idx = (t * torch.ones([image_a.size(0)], device=self.device)).long()
                z = torch.randn([image_a.size(0), 4*self.hparams.expended_channels], device=self.device)
                fake_b = self.g(xt, time_idx, z)
                # fake_bs.append(Xt_1.detach())
        # ! what the... 
        # fake_b = self.g(image_a)
        image_b = inversed_transform(image_b)
        fake_b = inversed_transform(fake_b)

        self.metric_fid.update(image_b, real=True)
        self.metric_fid.update(fake_b, real=False)

    def on_validation_epoch_end(self):
        self.metric_fid.compute()
        self.log("valid/fid", self.metric_fid, prog_bar=True)


if __name__ == "__main__":
    model = SbModel()
