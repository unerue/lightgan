import functools
import numpy as np
from typing import Callable, Optional
import torch
from torch import nn, FloatTensor
import torch.nn.functional as F
from torchvision import ops
import math
from .module_utils import initialize_weights



def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[
        :,
        :,
        max(-pad_y0, 0) : out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[3] - max(-pad_x1, 0),
    ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if len(k.shape) == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor**2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)



def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2**0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale



class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        # print("FusedLeakyReLU: ", input.abs().mean())
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        # print("FusedLeakyReLU: ", out.abs().mean())
        return out



class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()
        # print(in_dim, out_dim)
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (math.sqrt(1) / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input**2, dim=1, keepdim=True) + 1e-8)



class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = math.sqrt(1) / math.sqrt(in_channel * (kernel_size**2))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        # print("Before EqualConv2d: ", input.abs().mean())
        # !!!!!!! wrong
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # print("After EqualConv2d: ", out.abs().mean(), (self.weight * self.scale).abs().mean())

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class StyleGan2Encoder(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        no_antialias=False,
        stylegan2_num_downsamples: int = 1,
        load_size: int = 1024, 
        crop_size: int = 64
    ):
        super().__init__()
        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        blur_kernel = [1, 3, 3, 1]

        cur_res = 2 ** int((np.rint(np.log2(min(load_size, crop_size)))))
        convs = [nn.Identity(), ConvLayer(input_nc, channels[cur_res], 1)]

        num_downsampling = stylegan2_num_downsamples
        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res // 2]
            convs.append(
                # ResBlock
                StyledResBlock(in_channel, out_channel, blur_kernel, downsample=True)
            )
            cur_res = cur_res // 2

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(StyledResBlock(n_channel, n_channel, downsample=False))

        self.convs = nn.Sequential(*convs)

    def forward(self, input, layers=[], get_features=False):
        feat = input
        feats = []
        if -1 in layers:
            layers.append(len(self.convs) - 1)
        for layer_id, layer in enumerate(self.convs):
            feat = layer(feat)
            # print(layer_id, " features ", feat.abs().mean())
            if layer_id in layers:
                feats.append(feat)

        if get_features:
            return feat, feats
        else:
            return feat


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim=None,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size**2
        self.scale = math.sqrt(1) / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        if style_dim is not None and style_dim > 0:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if style is not None:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        else:
            style = torch.ones(batch, 1, in_channel, 1, 1).cuda()
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim=None,
        embedding_dim=None,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        inject_noise=True,
    ):
        super().__init__()

        self.inject_noise = inject_noise
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )
        # self.dense = EqualLinear(
        #     embedding_dim, out_channel, bias_init=0, activation="fused_lrelu"
        # )
        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, time_cond=None, noise=None):
        out = self.conv(input, style)
        if time_cond is not None:
            out += self.dense(time_cond)[:, :, None, None]
        if noise is not None:
            if self.inject_noise:
                out = self.noise(out, noise=noise)

        # out = out + self.bias
        out = self.activate(out)

        return out


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True, skip_gain=1.0):
        super().__init__()

        self.skip_gain = skip_gain
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel)

        if in_channel != out_channel or downsample:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, input):
        out = self.conv1(input)  # ! wrong
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out * self.skip_gain + skip) / math.sqrt(self.skip_gain ** 2 + 1.0)

        return out



class StyleGan2Decoder(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        no_antialias=False,
        num_downsamples: int = 1,
        load_size: int = 1024,
        crop_size: int = 64
    ):
        super().__init__()
        blur_kernel = [1, 3, 3, 1]

        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }
        # inject_noise = None
        cur_res = 2 ** int((np.rint(np.log2(min(load_size, crop_size))))) // (2**num_downsamples)
        convs = []
        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            # ResBlock
            convs.append(StyledResBlock(n_channel, n_channel, downsample=False))

        for i in range(num_downsamples):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res * 2]
            inject_noise = "small" not in "stylegan2"
            convs.append(
                StyledConv(in_channel, out_channel, 3, upsample=True, blur_kernel=blur_kernel, inject_noise=inject_noise)
            )
            cur_res = cur_res * 2

        convs.append(ConvLayer(channels[cur_res], 3, 1))
        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        out = self.convs(input)
        return out



def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class StyleGan2Generator(nn.Module):
    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=64,
        use_dropout=False,
        n_blocks=6,
        padding_type="reflect",
        no_antialias=False,
        lr: float = 0.0002,
        style_dim: int = None,
        n_mlp: int = 8,
        embedding_type: str = None
    ):
        super().__init__()
        self.ngf = ngf
        self.encoder = StyleGan2Encoder(
            input_nc,
            output_nc,
            ngf,
            use_dropout,
            n_blocks,
            padding_type,
            no_antialias,
        )
        self.decoder = StyleGan2Decoder(
            input_nc,
            output_nc,
            ngf,
            use_dropout,
            n_blocks,
            padding_type,
            no_antialias,
            num_downsamples=1
        )
        # print(self.decoder)
        # import sys
        # sys.exit()
        # layers = [PixelNorm()]
        # for i in range(self.opt.n_mlp):
        #     layers.append(
        #         EqualLinear(
        #             self.opt.style_dim,
        #             self.opt.style_dim,
        #             lr_mul=self.opt.lr * 0.01,
        #             activation="fused_lrelu",
        #         )
        #     )

        # self.style = nn.Sequential(*layers)

        # layers_time = []

        # layers_time.append(
        #     EqualLinear(
        #         ngf * 4, ngf * 4, activation="fused_lrelu", lr_mul=self.opt.lr * 0.01
        #     )
        # )
        # # modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        # # nn.init.zeros_(modules[-1].bias)
        # layers_time.append(EqualLinear(ngf * 4, ngf * 4, lr_mul=self.opt.lr * 0.01))
        # self.time_embedding = nn.Sequential(*layers_time)
        # # modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
        # # nn.init.zeros_(modules[-1].bias)

    def forward(self, input, layers=[], encode_only=False):
    # def forward(self, input, time_cond, z, layers=[], encode_only=False):
        # print(z.shape)
        # zemb = self.style(z)

        # if self.opt.embedding_type == "fourier":
        #     # Gaussian Fourier features embeddings.
        #     used_sigmas = time_cond
        #     temb = modules[m_idx](torch.log(used_sigmas))
        #     m_idx += 1

        # elif self.opt.embedding_type == "positional":
        #     # Sinusoidal positional embeddings.
        #     timesteps = time_cond
        #     temb = get_timestep_embedding(timesteps, self.ngf * 4)
        #     # print(temb.shape)
        # else:
        #     raise ValueError(f"embedding type {self.embedding_type} unknown.")
        # temb = self.time_embedding(temb)
        # # print(temb.shape)
        feat, feats = self.encoder(input, layers, True)
        # print(feat.shape)
        # if len(feats) > 0:
        #     # print(feats[0].shape)
        if encode_only:
            return feats
        else:
            # print(temb.shape)
            # fake = self.decoder(feat, zemb, temb)
            fake = self.decoder(feat)

            if len(layers) > 0:
                return fake, feats
            else:
                return fake
            

class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            EqualLinear(
                embedding_dim, hidden_dim, bias_init=0, activation="fused_lrelu"
            ),
            EqualLinear(hidden_dim, output_dim, bias_init=0, activation="fused_lrelu"),
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb


class ResBlock_cond(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        t_emb_dim,
        blur_kernel=[1, 3, 3, 1],
        downsample=True,
        skip_gain=1.0,
        residual=True,
    ):
        super().__init__()

        self.skip_gain = skip_gain
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(
            in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel
        )
        # self.residual = residual
        if in_channel != out_channel or downsample:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                activate=False,
                bias=False,
            )
        else:
            self.skip = nn.Identity()

        # self.dense = EqualLinear(t_emb_dim, in_channel, bias_init=0)

    # def forward(self, input, t_emb):
    def forward(self, input):
        out = self.conv1(input)
        # out += self.dense(t_emb)[..., None, None]
        out = self.conv2(out)
        # if self.residual:
        #     skip = self.skip(input)
        #     out = (out * self.skip_gain + skip) / math.sqrt(self.skip_gain**2 + 1.0)
        # # else:
        # #     out = out
        skip = self.skip(input)
        out = (out * self.skip_gain + skip) / math.sqrt(self.skip_gain ** 2 + 1.0)
        return out


class StyleGan2Discriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf=64,
        n_layers=3,
        t_emb_dim=4 * 64,
        no_antialias=False,
        size=None,
        load_size: int = 286,
        crop_size: int = 256,
        **kwagrs,
    ):
        super().__init__()
        self.stddev_group = 16
        if size is None:
            size = 2 ** int((np.rint(np.log2(min(load_size, crop_size)))))
            if "patch" in "stylegan2" and self.opt.D_patch_size is not None:
                size = 2 ** int(np.log2(self.opt.D_patch_size))

        blur_kernel = [1, 3, 3, 1]
        channel_multiplier = ndf / 64
        channels = {
            4: min(384, int(4096 * channel_multiplier)),
            8: min(384, int(2048 * channel_multiplier)),
            16: min(384, int(1024 * channel_multiplier)),
            32: min(384, int(512 * channel_multiplier)),
            64: int(256 * channel_multiplier),
            128: int(128 * channel_multiplier),
            256: int(64 * channel_multiplier),
            512: int(32 * channel_multiplier),
            1024: int(16 * channel_multiplier),
        }

        convs = [ConvLayer(input_nc, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]
        # ['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2']
        if "smallpatch" in "stylegan2":
            final_res_log2 = 4
        elif "patch" in "stylegan2":
            final_res_log2 = 3
        else:
            final_res_log2 = 2
        # self.convs_init = nn.Sequential(*convs)
        # self.convs = nn.ModuleList()
        for i in range(log_size, final_res_log2, -1):
            out_channel = channels[2 ** (i - 1)]

            # self.convs.append(
            #     ResBlock_cond(
            #         in_channel, out_channel, t_emb_dim, blur_kernel, residual=False
            #     )
            # )
            # ResBlock
            convs.append(StyledResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel
        self.convs = nn.Sequential(*convs)
        # ['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2']
        if False and "tile" in "stylegan2":
            in_channel += 1
        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        # ['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2']
        if "patch" in "stylegan2":
            self.final_linear = ConvLayer(channels[4], 1, 3, bias=False, activate=False)
        else:
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
                EqualLinear(channels[4], 1),
            )
        # self.t_embed = TimestepEmbedding(
        #     embedding_dim=t_emb_dim,
        #     hidden_dim=t_emb_dim,
        #     output_dim=t_emb_dim,
        #     act=nn.LeakyReLU(0.2),
        # )

    def forward(self, input, get_minibatch_features=False):
        # t_embed = self.t_embed(t)
        # input_x = torch.cat((input, x_t), dim = 1)
        # ['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'],
        if "patch" in "stylegan2" and self.opt.D_patch_size is not None:
            h, w = input.size(2), input.size(3)
            y = torch.randint(h - self.opt.D_patch_size, ())
            x = torch.randint(w - self.opt.D_patch_size, ())
            input_x = input_x[
                :, :, y : y + self.opt.D_patch_size, x : x + self.opt.D_patch_size
            ]
        out = input
        # out = self.convs_init(out)
        for i, conv in enumerate(self.convs):
            # out = conv(out, t_embed)
            out = conv(out)
            # print(i, out.abs().mean())
        # out = self.convs(input)

        batch, channel, height, width = out.shape

        # tilestylegan2 
        if False and "tile" in "stylegan2":
            group = min(batch, self.stddev_group)
            stddev = out.view(group, -1, 1, channel // 1, height, width)
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        # print(out.abs().mean())

        # ['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2']
        if "patch" not in "stylegan2":
            out = out.view(batch, -1)
        out = self.final_linear(out)

        return out