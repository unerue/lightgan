import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import functools
import math
from torch.nn import init

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt

class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.style_net = nn.Linear(style_dim, in_channel * 2)

        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input, style):
        
        style = self.style_net(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = gamma * input + beta

        return out

class ResnetBlock_cond(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_cond, self).__init__()
        self.conv_block,self.adaptive,self.conv_fin = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,temb_dim,z_dim):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        
        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == 'reflect':
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        self.conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        self.adaptive = AdaptiveLayer(dim,z_dim) 
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
        
        self.Dense_time = nn.Linear(temb_dim, dim)
        # self.Dense_time.weight.data = default_init()(self.Dense_time.weight.data.shape)
        nn.init.zeros_(self.Dense_time.bias)
        
        self.style = nn.Linear(z_dim, dim * 2)

        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0
        
        return self.conv_block,self.adaptive,self.conv_fin

    def forward(self, x,time_cond,z):
        
        time_input = self.Dense_time(time_cond)
        for n,layer in enumerate(self.conv_block):
            out = layer(x)
            if n==0:
                out += time_input[:, :, None, None]
        out = self.adaptive(out,z)
        for layer in self.conv_fin:
            out = layer(out)
        """Forward function (with skip connections)"""
        out = x + out  # add skip connections
        return out
    

class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
      # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class ResnetGenerator_ncsn(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(
        self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
        padding_type='reflect',  no_antialias=False, no_antialias_up=False, n_mlp=3
    ):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_ncsn, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        self.ngf = ngf
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)
                          # nn.AvgPool2d(kernel_size=2, stride=2)
                        ]
        self.model_res = nn.ModuleList()
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            self.model_res += [ResnetBlock_cond(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias,temb_dim=4*ngf,z_dim=4*ngf)]

        model_upsample = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_upsample += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model_upsample += [
                    Upsample(ngf * mult),
                    # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model_upsample += [nn.ReflectionPad2d(3)]
        model_upsample += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_upsample += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_upsample = nn.Sequential(*model_upsample)
        mapping_layers = [PixelNorm(),
                      nn.Linear(self.ngf*4, self.ngf*4),
                      nn.LeakyReLU(0.2)]
        for _ in range(n_mlp):
            mapping_layers.append(nn.Linear(self.ngf*4, self.ngf*4))
            mapping_layers.append(nn.LeakyReLU(0.2))
        self.z_transform = nn.Sequential(*mapping_layers)
        modules_emb = []
        modules_emb += [nn.Linear(self.ngf,self.ngf*4)]
        
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        modules_emb += [nn.Linear(self.ngf*4,self.ngf*4)]
        
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb += [nn.LeakyReLU(0.2)]
        self.time_embed = nn.Sequential(*modules_emb)
        
    def forward(self, x, time_cond,z,layers=[], encode_only=False):
        z_embed = self.z_transform(z)
        # print(z_embed.shape)
        temb = get_timestep_embedding(time_cond, self.ngf)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                
            for layer_id, layer in enumerate(self.model_res):
                feat = layer(feat,time_embed,z_embed)
                if layer_id+len(self.model) in layers:
                    feats.append(feat)
                if layer_id+len(self.model) == layers[-1] and encode_only:
                    return feats
            return feat, feats
        else:
            
            out = self.model(x)
            for layer in self.model_res:
                out = layer(out,time_embed,z_embed)
            out = self.model_upsample(out)
            return out


class ConvBlock_cond(nn.Module):
    def __init__(self, in_channel, out_channel,t_emb_dim, kernel_size=4,stride=1,padding=1,norm_layer=None,downsample=True,use_bias=None):
        super().__init__()
        self.downsample=downsample
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)
        
        if norm_layer is not None:
            self.use_norm =True
            self.norm = norm_layer(out_channel)
        else:
            self.use_norm = False
        self.act = nn.LeakyReLU(0.2, True)
        self.down = Downsample(out_channel)
        
        self.dense= nn.Linear(t_emb_dim, out_channel)
    def forward(self, input,t_emb):
        out = self.conv1(input)
        out += self.dense(t_emb)[..., None, None]
        if self.use_norm:
            out = self.norm(out)
        out = self.act(out)
        if self.downsample:
            out = self.down(out)
        
        return out


class TimestepEmbedding(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
            # EqualLinear(hidden_dim, output_dim,bias_init = 0, activation='fused_lrelu')
        )

    def forward(self, temp):
        temb = get_timestep_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

class NLayerDiscriminator_ncsn(nn.Module):
    """Defines a PatchGAN discriminator
    Construct a PatchGAN discriminator

    Parameters:
        input_nc (int)  -- the number of channels in input images
        ndf (int)       -- the number of filters in the last conv layer
        n_layers (int)  -- the number of conv layers in the discriminator
        norm_layer      -- normalization layer
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.model_main = []
        kw = 4
        padw = 1

        self.t_embed = TimestepEmbedding(
            embedding_dim=4*ndf,
            hidden_dim=4*ndf,
            output_dim=4*ndf,
            act=nn.LeakyReLU(0.2),
        )
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            self.model_main.append(ConvBlock_cond(input_nc, ndf, 4*ndf,kernel_size=kw, stride=1, padding=padw,use_bias=use_bias))

        nf_mult = 1
        nf_mult_prev = 1
         
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            else:
                self.model_main.append(
                    ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult, 4*ndf,kernel_size=kw, stride=1, padding=padw,use_bias=use_bias,norm_layer=norm_layer)
                    
                )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.model_main.append(
            ConvBlock_cond(ndf * nf_mult_prev, ndf * nf_mult,4*ndf, kernel_size=kw, stride=1, padding=padw,use_bias=use_bias,norm_layer=norm_layer,downsample=False)
            
        )
        self.final_conv =nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        self.model_main = nn.Sequential(*self.model_main)

    def forward(self, input, t_emb, input2=None):
        """Standard forward."""
        t_emb = self.t_embed(t_emb)
        if input2 is not None:
            out = torch.cat([input, input2], dim=1)
        else:
            
            out = input
        for layer in self.model_main:
            out = layer(out,t_emb)
            
        return self.final_conv(out)
