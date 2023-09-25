import functools
import numpy as np
from typing import Callable, Optional
import torch
from torch import nn, FloatTensor
import torch.nn.functional as F
from torchvision import ops
import math


filer_sizes = {
    1: [1.0],
    2: [1.0, 1.0],
    3: [1.0, 2.0, 1.0],
    4: [1.0, 3.0, 3.0, 1.0],
    5: [1.0, 4.0, 6.0, 4.0, 1.0],
    6: [1.0, 5.0, 10.0, 10.0, 5.0, 1.0],
    7: [1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0],
}


def get_filter(filt_size: int = 3):
    a = torch.Tensor(filer_sizes[filt_size])
    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)
    return filt


def get_pad_layer(pad_type):
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        print("Pad type [%s] not recognized" % pad_type)
    return PadLayer


class Downsample(nn.Module):
    def __init__(
        self,
        channels,
        pad_type="reflect",
        filt_size=3,
        stride=2,
        pad_off=0
    ) -> None:
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [
            int(1.0 * (filt_size - 1) / 2),
            int(math.ceil(1.0 * (filt_size - 1) / 2)),
            int(1.0 * (filt_size - 1) / 2),
            int(math.ceil(1.0 * (filt_size - 1) / 2)),
        ]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )
        self.pad_layer = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad_layer(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(
                self.pad_layer(inp), self.filt, stride=self.stride, groups=inp.shape[1]
            )


class Upsample(nn.Module):
    def __init__(
        self,
        channels,
        pad_type="repl",
        filt_size=4,
        stride=2
    ) -> None:
        super().__init__()
        self.filt_size = filt_size
        # self.filt_odd = np.mod(filt_size, 2) == 1
        self.filt_odd = np.mod(filt_size, 2) == 1
        # self.filt_odd = filt_size % 2 == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.0)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer(
            "filt", filt[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        self.pad_layer = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(
            self.pad_layer(inp),
            self.filt,
            stride=self.stride,
            padding=1 + self.pad_size,
            groups=inp.shape[1],
        )[:, :, 1:, 1:]
        if self.filt_odd:
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class Conv2dPadBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        pad_layer: Callable[..., nn.Module] = nn.ZeroPad2d,
        use_bias: bool = True
    ) -> None:
        """
        input_dim:
        output_dim:
        kernel_size:
        stride:
        padding: 
        norm: default none | batch | instance | layer
        activation: default relu | lrelu | prelu | selu | tanh | none
        pad_type: default zero | reflect
        """
        super().__init__()
        if isinstance(pad_layer, (nn.ZeroPad2d, nn.ReflectionPad2d)):
            self.pad_layer = pad_layer(padding)
        if norm_layer is not None:
            if isinstance(norm_layer, (nn.BatchNorm2d, nn.InstanceNorm2d, LayerNorm)):
                self.norm_layer = norm_layer(output_dim)
            elif isinstance(norm_layer, nn.InstanceNorm2d):
                self.norm = norm_layer(output_dim, track_running_stats=False)
        if activation_layer is not None:
            if isinstance(activation_layer, (nn.ReLU, nn.LeakyReLU, nn.SELU)):
                self.activation = activation_layer(inplace=True)
            elif isinstance(activation_layer, (nn.PReLU, nn.Tanh)):
                self.activation_layer = activation_layer

        self.conv = ops.Conv2dNormActivation(
            input_dim, output_dim, kernel_size, stride, padding,
            norm_layer=norm_layer, activation_layer=activation_layer, bias=use_bias
        )

    def forward(self, x):
        return self.conv(self.pad_layer(x))


class ResBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.InstanceNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        pad_layer: Callable[..., nn.Module] = nn.ZeroPad2d,
        nz: int = 0
    ) -> None:
        super(ResBlock, self).__init__()

        model = []
        model += [
            Conv2dPadBlock(
                dim + nz,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                pad_layer=pad_layer,
            )
        ]
        model += [
            Conv2dPadBlock(
                dim,
                dim + nz,
                kernel_size=3,
                stride=1,
                padding=1,
                norm_layer=norm_layer,
                activation_layer=None,
                pad_layer=pad_layer
            )
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ResBlocks(nn.Module):
    def __init__(
        self,
        num_blocks,
        dim,
        norm_layer=nn.InstanceNorm2d,
        activation_layer=nn.ReLU,
        pad_layer=nn.ZeroPad2d,
        nz=0
    ):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [
                ResBlock(
                    dim,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                    pad_layer=pad_layer,
                    nz=nz
                )
            ]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResnetBlock(nn.Module):
    """
    Define a Resnet block
    Initialize the Resnet block

    A resnet block is a conv block with skip connections
    We construct a conv block with build_conv_block function,
    and implement skip connections in <forward> function.
    Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
    """
    def __init__(
        self,
        dim: int,
        padding_type,
        pad_layer: Callable[..., nn.Module] = nn.ReflectionPad2d,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = True,
        use_bias: bool = True
    ) -> None:
        super().__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        # if padding_type == "reflect":
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == "replicate":
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == "zero":
        #     p = 1
        # else:
        #     raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        # conv_block += [
        #     nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
        #     norm_layer(dim),
        #     nn.ReLU(True),
        # ]
        conv_block += [
            ops.Conv2dNormActivation(
                dim, dim, kernel_size=3, padding=0, 
                norm_layer=norm_layer, activation_layer=nn.ReLU,
                inplace=True, bias=use_bias
            ),
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        # if padding_type == "reflect":
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == "replicate":
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == "zero":
        #     p = 1
        # else:
        #     raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=use_bias),
            # norm_layer(dim),
            nn.InstanceNorm2d(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    
    Args:
        input_nc (int)      -- the number of channels in input images
        output_nc (int)     -- the number of channels in output images
        ngf (int)           -- the number of filters in the last conv layer
        norm_layer          -- normalization layer
        use_dropout (bool)  -- if use dropout layers
        n_blocks (int)      -- the number of ResNet blocks
        padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
    """
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        num_blocks=6,
        padding_type="reflect",
        no_antialias=False,
        no_antialias_up=False,
        use_bias=True
    ) -> None:
        super().__init__()
        assert num_blocks >= 0

        model = [
            nn.ReflectionPad2d(padding=3),
            ops.Conv2dNormActivation(
                input_nc,
                ngf,
                kernel_size=7,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU,
                inplace=True,
                bias=use_bias,
            )
        ]
        num_downsamples = 2
        for i in range(num_downsamples):
            mult = 2 ** i
            if no_antialias:
                model += [
                    ops.Conv2dNormActivation(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        norm_layer=norm_layer,
                        activation_layer=nn.ReLU,
                        inplace=True,
                        bias=use_bias,
                    )
                ]
            else:
                model += [
                    ops.Conv2dNormActivation(
                        ngf * mult,
                        ngf * mult * 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_layer=norm_layer,
                        activation_layer=nn.ReLU,
                        inplace=True,
                        bias=use_bias,
                    ),
                    Downsample(ngf * mult * 2),
                ]

        mult = num_downsamples ** 2
        for i in range(num_blocks):  # add ResNet blocks
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    norm_layer=norm_layer,
                    use_dropout=use_dropout,
                    use_bias=use_bias,
                )
            ]
        for i in range(num_downsamples):  # add upsampling layers
            mult = 2 ** (num_downsamples - i)
            if no_antialias_up:
                model += [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
            else:
                model += [
                    Upsample(ngf * mult),
                    ops.Conv2dNormActivation(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_layer=norm_layer,
                        activation_layer=nn.ReLU,
                        inplace=True,
                        bias=use_bias,
                    )
                ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, layers=[], encode_only=False):
        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            return fake


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""
    def __init__(
        self,
        input_nc: int,
        ndf: int = 64,
        num_layers: int = 3,
        norm_layer: Callable[..., nn.Module]=nn.BatchNorm2d,
        no_antialias=False,
    ):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True),
            ]
        else:
            sequence = [
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw),
                nn.LeakyReLU(0.2, True),
                Downsample(ndf),
            ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                ]
            else:
                sequence += [
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=1,
                        padding=padw,
                        bias=use_bias,
                    ),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult),
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**num_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    return net


class Normalize(nn.Module):
    """CUT"""
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class PatchSampleF(nn.Module):
    """CUT"""
    def __init__(
        self,
        use_mlp=False,
        init_type='normal',
        init_gain=0.02,
        nc=256,
        gpu_ids=[]
    ) -> None:
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super().__init__()
        self.l2norm = Normalize(2)
        self.use_mlp = use_mlp
        self.nc = nc  # hard-coded
        self.mlp_init = False
        self.init_type = init_type
        self.init_gain = init_gain
        self.gpu_ids = gpu_ids

    def create_mlp(self, feats):
        for mlp_id, feat in enumerate(feats):
            input_nc = feat.shape[1]
            mlp = nn.Sequential(*[
                nn.Linear(input_nc, self.nc),
                nn.ReLU(),
                nn.Linear(self.nc, self.nc)
            ])
            # if len(self.gpu_ids) > 0:
            #     mlp.cuda()
            setattr(self, 'mlp_%d' % mlp_id, mlp)
        init_net(self, self.init_type, self.init_gain, self.gpu_ids)
        self.mlp_init = True

    def forward(self, feats, num_patches=64, patch_ids=None):
        return_ids = []
        return_feats = []
        if self.use_mlp and not self.mlp_init:
            self.create_mlp(feats)
        for feat_id, feat in enumerate(feats):
            B, H, W = feat.shape[0], feat.shape[2], feat.shape[3]
            feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
            if num_patches > 0:
                if patch_ids is not None:
                    patch_id = patch_ids[feat_id]
                else:
                    # torch.randperm produces cudaErrorIllegalAddress for newer versions of PyTorch. https://github.com/taesungp/contrastive-unpaired-translation/issues/83
                    #patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
                    patch_id = np.random.permutation(feat_reshape.shape[1])
                    patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
                patch_id = torch.tensor(patch_id, dtype=torch.long, device=feat.device)
                x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
            else:
                x_sample = feat_reshape
                patch_id = []
            if self.use_mlp:
                mlp = getattr(self, 'mlp_%d' % feat_id)
                x_sample = mlp(x_sample)
            return_ids.append(patch_id)
            x_sample = self.l2norm(x_sample)

            if num_patches == 0:
                x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W])
            return_feats.append(x_sample)
        return return_feats, return_ids
