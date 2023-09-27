import itertools
from typing import Any, Optional
from enum import IntEnum

import torch
from torch import nn, optim
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

from ..ops import ResnetGenerator, NLayerDiscriminator, ImagePool


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
        self.y = ResnetGenerator(
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


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # basic_cond
        self.x = NLayerDiscriminator(
            input_nc=3,
            ndf=64,
            num_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=True,
        )
        self.y = NLayerDiscriminator(
            input_nc=3,
            ndf=64,
            num_layers=3,
            norm_layer=nn.InstanceNorm2d,
            no_antialias=True,
        )


class Initializer(IntEnum):
    NORMAL = 0
    XAVIER_NORMAL = 1
    KAIMING_NORMAL = 2
    ORTHOGONAL = 3


def initialize_weights(m: nn.Module, init_type: Initializer = 1, gain: float = 0.02):
    """
    define the initialization function
    init_type: normal | xavier normal | kaiming normal | orthogonal
    gain `` default to 0.02
    """
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
        if Initializer.NORMAL == init_type:
            nn.init.normal_(m.weight.data, mean=0.0, std=gain)
        elif Initializer.XAVIER_NORMAL == init_type:
            nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif Initializer.KAIMING_NORMAL == init_type:
            nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif Initializer.ORTHOGONAL == init_type:
            nn.init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=gain)
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


class CycleGanModel(LightningModule):
    """
    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    Args:
        input_channels: default=3
        expanded_channels: default=64


    For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
    A (source domain), B (target domain).
    Generators: G_A: A -> B; G_B: B -> A.
    Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
    Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
    Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
    Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
    Dropout is not used in the original CycleGAN paper.

    lambda_a: default=10.0 weight for cycle loss (A -> B -> A)
    lambda_b: default=10.0 weight for cycle loss (B -> A -> B)
    lambda_i: default=0.5 use identity mapping. 
        Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. 
        For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, 
        please set lambda_identity = 0.1

    lambda_gan: default=1.0  weight for GAN loss：GAN(G(X))
    lambda_nce: default=1.0  weight for NCE loss: NCE(G(X), X)
    lambda_sb: default=0.1  weight for SB loss
    nce_idt: type= default= False, nargs=?, const=True  use NCE loss for identity mapping: NCE(G(Y), Y))
    nce_layers: default=0,4,8,12,16  compute NCE loss on which layers
        nce_includes_all_negatives_from_minibatch: type= default= False, nargs=?, const=True (used for single image translation) 
        If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. 
        please see models/patchnce.py for more details.
    netF: default=mlp_sample | sample, reshape, mlp_sample  how to downsample the feature map
    netF_nc: default=256  number of channels for netF
    nce_T: default=0.07  temperature for NCE loss
    lmda: default=0.1  weight for SB loss
    num_patches: default=256  number of patches per layer
        flip_equivariance: default= False, nargs=?, const=True  Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT
        pool_size: default=0  no image pooling

    input_nc: default=3
    output_nc: default=3
    ngf: default=64  of gen filters in first conv layer
    num_timesteps: default=5  of discrim filters in the first conv layer

    embedding_type: default=positional | fourier, positional
    emdedding_dim: default=512,
    n_layers_D: default=3  only used if netD==n_layers
    style_dim: default=512  only used if netD==n_layers
    n_mlp: default=3  only used if netD==n_layers
    normG: default=instance | instance, batch, none  instance normalization or batch normalization for G
    normD: default=instance | instance, batch, none  instance normalization or batch normalization for D
    init_type: default=xavier | normal, xavier, kaiming, orthogonal  network initialization
    init_gain: default=0.02  scaling factor for normal, xavier and orthogonal.
    no_dropout
    std: default=0.25  scale of gaussian noise added to data
    tau: default=0.01  entropy parameter
    no_antialias: default=False  if specified, use stride=2 conv instead of antialiased-downsampling (sad)
    no_antialias_up: default=False  if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]
    dataset_mode: default=unaligned | unaligned, aligned, single, colorization
    stylegna2_G_num_downsampling: default=1  number of downsampling layers in stylegan2 generator
    gpu_ids: default=0
    gan_mode: default=lsgan | vanilla, lsgan | wgangp
    lr: default=0.0002  initial learning rate for adam
    beta1: default=0.5  momentum term of adam
    beta2: default=0.999  momentum term of adam
    pool_size: default=50  the size of image buffer that stores previously generated images
    lr_policy: default=linear | linear, step, plateau, cosine
    lr_decay_iters: default=50  multiply by a gamma every lr_decay_iters iterations
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

        # fid = FrechetInceptionDistance(feature=64)
        # kid = KernelInceptionDistance(feature=64)

        image_shape = (3, 256, 256)
        self.example_input_array = (
            torch.zeros((1, *image_shape)), torch.zeros((1, *image_shape))
        )
        self.training_step_outputs = []
        self.training_step_outputs2 = []

    def forward(self, x, z):
        image_a, image_b = x, z
        return self.g.x(image_a), self.g.y(image_b), self.d.x(image_a), self.d.y(image_b)

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