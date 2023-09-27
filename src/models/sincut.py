from typing import Any, Optional
import torch
from torch import nn, Tensor, optim
from lightning import LightningModule
from ..ops import (
    StyleGan2Generator,
    StyleGan2Discriminator,
    NLayerDiscriminator,
    ImagePool,
    PatchSampleF,
    GanLoss,
    PatchNceLoss,
    initialize_weights,
    Initializer
)

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class SinCutModel(LightningModule):
    """This class implements the single image translation model (Fig 9) of
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    lambdar1 = 1.0
    lambda_identity = 1.0

    nce_includes_all_negatives_from_minibatch = True
    dataset_mode = singleimage
    net_g = stylegan2
    gan_mode = nonsaturating
    num_patches = 1
    nce_layers = 0,2,4
    lambda_nce = 4.0
    ngf = 10
    ndf = 8
    lr = 0.002
    beta1 = 0.0
    beta2 = 0.99
    load_size = 1024
    crop_size = 64
    preprocess = zoom_and_patch

    preprocess zoom_and_patch
    batch_size = 16

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
        lambda_i: float = 1.0,
        lambda_nce: float = 4.0,  # fastcut 10.0
        lambda_gan: float = 1.0,
        nce_idt: bool = True,  # fastcut False
        pool_size: int = 0,
        batch_size: int = 1,
        num_patches: int = 1,
        nce_includes_all_negatives_from_minibatch: bool = True,
        lambda_r1: float = 1.0,
        nce_layers = [0, 2, 4],
        **kwargs
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lambda_a = 10.0
        self.g = StyleGan2Generator(
            input_nc=3,
            output_nc=3,
            ngf=10,
            use_dropout=True,
            n_blocks=6,
            padding_type="reflect",
            no_antialias=False,
            lr=0.002,
            # style_dim=,
            # n_mlp=,
        )
        self.f = PatchSampleF(
            use_mlp=True,
            init_type="normal",
            nc=256,
        )
        self.d = StyleGan2Discriminator(
            input_nc=3,
            ndf=8,
            n_layers=3,
            t_emb_dim=4*64,
            no_antialias=False,
            size=None,
            load_size=1024,
            crop_size=64,
        )

        self.g.apply(initialize_weights)
        # self.f.apply(initialize_weights)
        self.d.apply(initialize_weights)

        # self.fake_pool_a = ImagePool(0)
        # self.fake_pool_b = ImagePool(0)

        self.criterion1 = GanLoss("nonsaturating")
        # self.nce_layers = [0, 4, 8, 12, 16]
        self.nce_layers = [0, 2, 4]
        self.criterion_nce = []
        for _ in self.nce_layers:
            self.criterion_nce.append(PatchNceLoss(batch_size))

        image_shape = (3, 64, 64)
        # self.example_input_array = torch.zeros((16, *image_shape))
        self.training_step_outputs = []
        self.training_step_outputs2 = []


    # def forward(self, x):
    #     image_a = x
    #     return self.g(image_a)

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

        loss_idt = torch.nn.functional.l1_loss(idt_b, image_b) * self.hparams.lambda_i
        gen_loss = gen_loss + loss_idt
        self.log("gen_loss", gen_loss, prog_bar=True)
        return gen_loss

    def r1_loss(self, real_pred, real_img):
        # print(real_pred.shape, real_img.shape)
        # print(real_pred.device, real_img.device)
        # print(real_pred.requires_grad, real_img.requires_grad)
        grad_real, = torch.autograd.grad(
            outputs=real_pred.sum(),
            inputs=real_img,
            create_graph=True,
            retain_graph=True,
        )
        # print(grad_real.shape)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty * (self.hparams.lambda_r1 * 0.5)

    def discriminator_training_step(self, image_a, image_b):
        fake_b = self.fake_b.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake_b = self.d(fake_b)
        loss_d_fake = self.criterion1(pred_fake_b, False).mean()
        # real
        self.pred_real = self.d(image_b)
        loss_d_real = self.criterion1(self.pred_real, True).mean()

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

        # self.untoggle_optimizer(optimizer_d)
        # self.d.freeze()
        # for param in self.d.parameters():
        #     if not param.requires_grad:
        #         print("requires_grad=False")
        #         break

        set_requires_grad(self.d, True)
        self.toggle_optimizer(optimizer_d)
        image_b = image_b.requires_grad_()
        optimizer_d.zero_grad()
        d_loss = self.discriminator_training_step(image_a, image_b)
        loss_d_r1 = self.r1_loss(self.pred_real, image_b)
        self.log("r1_loss", loss_d_r1, prog_bar=True)
        d_loss = d_loss + loss_d_r1
        self.manual_backward(d_loss)
        optimizer_d.step()
        # scheduler_d.step()
        self.untoggle_optimizer(optimizer_d)

        # ! check self.toggle_optimizer(optimizer_g) or LitModel.freeze()
        self.toggle_optimizer(optimizer_g)
        set_requires_grad(self.d, False)
        # for param in self.d.parameters():
        #     if not param.requires_grad:
        #         print("requires_grad=False")
        #         break
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
    model = SinCutModel()