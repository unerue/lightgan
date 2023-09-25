import numpy as np
import torch

import util.util as util

from . import networks
from .base_model import BaseModel
from .patchnce import PatchNCELoss


class SbModel(BaseModel):
    """
    mode: sb | FastCUT, fastcut, sb
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
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configures options specific for SB model"""
        parser.add_argument(
            "--nce_idt",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="use NCE loss for identity mapping: NCE(G(Y), Y))",
        )
        parser.add_argument(
            "--nce_includes_all_negatives_from_minibatch",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.",
        )
        parser.add_argument(
            "--flip_equivariance",
            type=util.str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT",
        )
        # Set default parameters for CUT and FastCUT
        if opt.mode.lower() == "sb":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False,
                lambda_NCE=10.0,
                flip_equivariance=True,
                n_epochs=150,
                n_epochs_decay=50,
            )

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["G_GAN", "D_real", "D_fake", "G", "NCE", "SB"]
        self.visual_names = ["real_A", "real_A_noisy", "fake_B", "real_B"]
        if self.opt.phase == "test":
            self.visual_names = ["real"]
            for NFE in range(self.opt.num_timesteps):
                fake_name = "fake_" + str(NFE + 1)
                self.visual_names.append(fake_name)
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(",")]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ["NCE_Y"]
            self.visual_names += ["idt_B"]

        if self.isTrain:
            self.model_names = ["G", "F", "D", "E"]
        else:  # during test time, only load G
            self.model_names = ["G"]
        """
        input_nc: default=3
        output_nc: default=3
        ngf: default=64  of gen filters in first conv layer
        num_timesteps: default=5  of discrim filters in the first conv layer

        netD: default=basic_cond | basic, n_layers, pixel, patch, tilestylegan2, stylegan2
        netE: default=basic_cond | basic, n_layers, pixel, patch, tilestylegan2, stylegan2, patchstylegan2
        netG: default=resnet_9blocks_cond | resnet_9blocks, resnet_6blocks, unet_256, unet_128, stylegan2, smallstylegan2, resnet_cat
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
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(
            opt.input_nc,
            opt.output_nc,
            opt.ngf,
            opt.netG,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            opt.no_antialias_up,
            self.gpu_ids,
            opt,
        )
        self.netF = networks.define_F(
            opt.input_nc,
            opt.netF,
            opt.normG,
            not opt.no_dropout,
            opt.init_type,
            opt.init_gain,
            opt.no_antialias,
            self.gpu_ids,
            opt,
        )

        if self.isTrain:
            self.netD = networks.define_D(
                opt.output_nc,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                self.gpu_ids,
                opt,
            )
            self.netE = networks.define_D(
                opt.output_nc * 4,
                opt.ndf,
                opt.netD,
                opt.n_layers_D,
                opt.normD,
                opt.init_type,
                opt.init_gain,
                opt.no_antialias,
                self.gpu_ids,
                opt,
            )
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_E = torch.optim.Adam(
                self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_E)

    def data_dependent_initialize(self, data, data2):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, data2)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()  # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_G_loss().backward()
            self.compute_D_loss().backward()
            self.compute_E_loss().backward()
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(
                    self.netF.parameters(),
                    lr=self.opt.lr,
                    betas=(self.opt.beta1, self.opt.beta2),
                )
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)

        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()

    def set_input(self, input, input2=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == "AtoB"
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        if input2 is not None:
            self.real_A2 = input2["A" if AtoB else "B"].to(self.device)
            self.real_B2 = input2["B" if AtoB else "A"].to(self.device)

        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        tau = self.opt.tau
        T = self.opt.num_timesteps
        incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        times = torch.tensor(times).float().cuda()
        self.times = times
        bs = self.real_A.size(0)
        time_idx = (
            torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()
        ).long()
        self.time_idx = time_idx
        self.timestep = times[time_idx]

        with torch.no_grad():
            self.netG.eval()
            for t in range(self.time_idx.int().item() + 1):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).reshape(-1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                Xt = (
                    self.real_A
                    if (t == 0)
                    else (1 - inter) * Xt
                    + inter * Xt_1.detach()
                    + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
                )
                time_idx = (
                    t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)
                ).long()
                time = times[time_idx]
                z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(
                    self.real_A.device
                )
                Xt_1 = self.netG(Xt, time_idx, z)

                Xt2 = (
                    self.real_A2
                    if (t == 0)
                    else (1 - inter) * Xt2
                    + inter * Xt_12.detach()
                    + (scale * tau).sqrt()
                    * torch.randn_like(Xt2).to(self.real_A.device)
                )
                time_idx = (
                    t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)
                ).long()
                time = times[time_idx]
                z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(
                    self.real_A.device
                )
                Xt_12 = self.netG(Xt2, time_idx, z)

                if self.opt.nce_idt:
                    XtB = (
                        self.real_B
                        if (t == 0)
                        else (1 - inter) * XtB
                        + inter * Xt_1B.detach()
                        + (scale * tau).sqrt()
                        * torch.randn_like(XtB).to(self.real_A.device)
                    )
                    time_idx = (
                        t
                        * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)
                    ).long()
                    time = times[time_idx]
                    z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(
                        self.real_A.device
                    )
                    Xt_1B = self.netG(XtB, time_idx, z)
            if self.opt.nce_idt:
                self.XtB = XtB.detach()
            self.real_A_noisy = Xt.detach()
            self.real_A_noisy2 = Xt2.detach()

        z_in = torch.randn(size=[2 * bs, 4 * self.opt.ngf]).to(self.real_A.device)
        z_in2 = torch.randn(size=[bs, 4 * self.opt.ngf]).to(self.real_A.device)
        """Run forward pass"""
        self.real = (
            torch.cat((self.real_A, self.real_B), dim=0)
            if self.opt.nce_idt and self.opt.isTrain
            else self.real_A
        )

        self.realt = (
            torch.cat((self.real_A_noisy, self.XtB), dim=0)
            if self.opt.nce_idt and self.opt.isTrain
            else self.real_A_noisy
        )

        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (
                np.random.random() < 0.5
            )
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.realt = torch.flip(self.realt, [3])

        self.fake = self.netG(self.realt, self.time_idx, z_in)
        self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2)
        self.fake_B = self.fake[: self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0) :]

        if self.opt.phase == "test":
            tau = self.opt.tau
            T = self.opt.num_timesteps
            incs = np.array([0] + [1 / (i + 1) for i in range(T - 1)])
            times = np.cumsum(incs)
            times = times / times[-1]
            times = 0.5 * times[-1] + 0.5 * times
            times = np.concatenate([np.zeros(1), times])
            times = torch.tensor(times).float().cuda()
            self.times = times
            bs = self.real.size(0)
            time_idx = (
                torch.randint(T, size=[1]).cuda() * torch.ones(size=[bs]).cuda()
            ).long()
            self.time_idx = time_idx
            self.timestep = times[time_idx]
            visuals = []
            with torch.no_grad():
                self.netG.eval()
                for t in range(self.opt.num_timesteps):
                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = (delta / denom).reshape(-1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).reshape(-1, 1, 1, 1)
                    Xt = (
                        self.real_A
                        if (t == 0)
                        else (1 - inter) * Xt
                        + inter * Xt_1.detach()
                        + (scale * tau).sqrt()
                        * torch.randn_like(Xt).to(self.real_A.device)
                    )
                    time_idx = (
                        t
                        * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)
                    ).long()
                    time = times[time_idx]
                    z = torch.randn(size=[self.real_A.shape[0], 4 * self.opt.ngf]).to(
                        self.real_A.device
                    )
                    Xt_1 = self.netG(Xt, time_idx, z)

                    setattr(self, "fake_" + str(t + 1), Xt_1)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        bs = self.real_A.size(0)

        fake = self.fake_B.detach()
        std = torch.rand(size=[1]).item() * self.opt.std

        pred_fake = self.netD(fake, self.time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        self.pred_real = self.netD(self.real_B, self.time_idx)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_E_loss(self):
        bs = self.real_A.size(0)

        """Calculate GAN loss for the discriminator"""

        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)
        temp = torch.logsumexp(
            self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0
        ).mean()
        self.loss_E = (
            -self.netE(XtXt_1, self.time_idx, XtXt_1).mean() + temp + temp**2
        )

        return self.loss_E

    def compute_G_loss(self):
        bs = self.real_A.size(0)
        tau = self.opt.tau

        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        std = torch.rand(size=[1]).item() * self.opt.std

        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            self.loss_G_GAN = (
                self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            )
        else:
            self.loss_G_GAN = 0.0
        self.loss_SB = 0
        if self.opt.lambda_SB > 0.0:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)

            bs = self.opt.batch_size

            ET_XY = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - torch.logsumexp(
                self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0
            )
            self.loss_SB = (
                -(self.opt.num_timesteps - self.time_idx[0])
                / self.opt.num_timesteps
                * self.opt.tau
                * ET_XY
            )
            self.loss_SB += self.opt.tau * torch.mean(
                (self.real_A_noisy - self.fake_B) ** 2
            )
        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + self.loss_SB + loss_NCE_both
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        z = torch.randn(size=[self.real_A.size(0), 4 * self.opt.ngf]).to(
            self.real_A.device
        )
        feat_q = self.netG(tgt, self.time_idx * 0, z, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.time_idx * 0, z, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(
            feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers
        ):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
