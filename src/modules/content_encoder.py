import functools
import math
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import Parameter as P

from diffusers import ModelMixin
from diffusers.configuration_utils import (ConfigMixin, 
                                           register_to_config)

from src.modules.attention import BasicTransformerBlock


def proj(x, y):
   return torch.mm(y, x.t()) * y / torch.mm(y, y.t())


def gram_schmidt(x, ys):
    for y in ys:
        x = x - proj(x, y)
    return x


def power_iteration(W, u_, update=True, eps=1e-12):
    us, vs, svs = [], [], []
    for i, u in enumerate(u_):
        with torch.no_grad():
            v = torch.matmul(u, W)
            v = F.normalize(gram_schmidt(v, vs), eps=eps)
            vs += [v]
            u = torch.matmul(v, W.t())
            u = F.normalize(gram_schmidt(u, us), eps=eps)
            us += [u]
            if update:
                u_[i][:] = u
        svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    return svs, us, vs


class LinearBlock(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        norm='none', 
        act='relu', 
        use_sn=False
    ):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)
        if use_sn:
            self.fc = nn.utils.spectral_norm(self.fc)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if act == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.activation = nn.Tanh()
        elif act == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(act)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(
        self, 
        nf_in, 
        nf_out, 
        nf_mlp, 
        num_blocks, 
        norm, 
        act, 
        use_sn =False
    ):
        super(MLP,self).__init__()
        self.model = nn.ModuleList()
        nf = nf_mlp
        self.model.append(LinearBlock(nf_in, nf, norm = norm, act = act, use_sn = use_sn))
        for _ in range((num_blocks - 2)):
            self.model.append(LinearBlock(nf, nf, norm=norm, act=act, use_sn=use_sn))
        self.model.append(LinearBlock(nf, nf_out, norm='none', act ='none', use_sn = use_sn))
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class SN(object):
    def __init__(
        self, 
        num_svs, 
        num_itrs, 
        num_outputs, 
        transpose=False, 
        eps=1e-12
    ):
        self.num_itrs = num_itrs
        self.num_svs = num_svs
        self.transpose = transpose
        self.eps = eps
        for i in range(self.num_svs):
            self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
            self.register_buffer('sv%d' % i, torch.ones(1))

    @property
    def u(self):
        return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

    @property
    def sv(self):
        return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]

    def W_(self):
        W_mat = self.weight.view(self.weight.size(0), -1)
        if self.transpose:
            W_mat = W_mat.t()
        for _ in range(self.num_itrs):
            svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps)
        if self.training:
            with torch.no_grad():
                for i, sv in enumerate(svs):
                    self.sv[i][:] = sv
        return self.weight / svs[0]

class SNConv2d(nn.Conv2d, SN):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                        padding, dilation, groups, bias)
        SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)

    def forward(self, x):
        return F.conv2d(x, self.W_(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward_wo_sn(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


class SNLinear(nn.Linear, SN):
    def __init__(self, in_features, out_features, bias=True,
                num_svs=1, num_itrs=1, eps=1e-12):
        nn.Linear.__init__(self, in_features, out_features, bias)
        SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)

    def forward(self, x):
        return F.linear(x, self.W_(), self.bias)


class Attention(nn.Module):
    def __init__(
        self, 
        ch, 
        which_conv=SNConv2d, 
        name='attention'
    ):
        super(Attention, self).__init__()
        self.ch = ch
        self.which_conv = which_conv
        self.theta = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = self.which_conv(self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = self.which_conv(self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = self.which_conv(self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        # Learnable gain parameter
        self.gamma = P(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2,2])
        g = F.max_pool2d(self.g(x), [2,2])
        
        theta = theta.view(-1, self. ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self. ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self. ch // 2, x.shape[2] * x.shape[3] // 4)
        
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        
        o = self.o(torch.bmm(g, beta.transpose(1,2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=SNConv2d, wide=True,
                preactivation=False, activation=None, downsample=None,):
        super(DBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels

        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv = which_conv
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)
    def shortcut(self, x):
        if self.preactivation:
            if self.learnable_sc:
                x = self.conv_sc(x)
            if self.downsample:
                x = self.downsample(x)
        else:
            if self.downsample:
                x = self.downsample(x)
            if self.learnable_sc:
                x = self.conv_sc(x)
        return x

    def forward(self, x):
        if self.preactivation:
            h = F.relu(x)
        else:
            h = x
        h = self.conv1(h)
        h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d,which_bn= nn.BatchNorm2d, activation=None,
                 upsample=None):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv,self.which_bn =which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(in_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    
    def forward(self, x):
        h = self.activation(self.bn1(x))
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = self.activation(self.bn2(h))
        h = self.conv2(h)
        if self.learnable_sc:
            x = self.conv_sc(x)
        return h + x


class GBlock2(nn.Module):
    def __init__(self, in_channels, out_channels,
                which_conv=nn.Conv2d, activation=None,
                upsample=None, skip_connection = True):
        super(GBlock2, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv = which_conv
        self.activation = activation
        self.upsample = upsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                            kernel_size=1, padding=0)

        # upsample layers
        self.upsample = upsample
        self.skip_connection = skip_connection

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        
        h = self.activation(h)
        h = self.conv2(h)
        
        if self.learnable_sc:
            x = self.conv_sc(x)


        if self.skip_connection:
            out = h + x
        else:
            out = h
        return out

def content_encoder_arch(ch =64,out_channel_multiplier = 1, input_nc = 3):
    arch = {}
    n=2
    arch[64] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2]],
                                'out_channels' : [item * ch for item in [1,2,4]],
                                'resolution': [40,20,10]}
    arch[80] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2]],
                                'out_channels' : [item * ch for item in [1,2,4]],
                                'resolution': [40,20,10]}
    arch[96] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2]],
                                'out_channels' : [item * ch for item in [1,2,4]],
                                'resolution': [48,24,12]}
                                
    arch[128] = {'in_channels':   [input_nc] + [ch*item for item in  [1,2,4,8]],
                                'out_channels' : [item * ch for item in [1,2,4,8,16]],
                                'resolution': [64,32,16,8,4]}
    
    arch[256] = {'in_channels':[input_nc]+[ch*item for item in [1,2,4,8,8]],
                                'out_channels':[item*ch for item in [1,2,4,8,8,16]],
                                'resolution': [128,64,32,16,8,4]}
    return arch

class ContentEncoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, G_ch=64, G_wide=True, resolution=128, 
                 G_kernel_size=3, G_attn='64_32_16_8', n_classes=1000, 
                 num_G_SVs=1, num_G_SV_itrs=1, G_activation=nn.ReLU(inplace=False), 
                 SN_eps=1e-12, output_dim=1,  G_fp16=False, 
                 G_init='N02',  G_param='SN', nf_mlp = 512, nEmbedding = 256, input_nc = 3,output_nc = 3):
        super(ContentEncoder, self).__init__()

        self.ch = G_ch
        self.G_wide = G_wide
        self.resolution = resolution
        self.kernel_size = G_kernel_size
        self.attention = G_attn
        self.n_classes = n_classes
        self.activation = G_activation
        self.init = G_init
        self.G_param = G_param
        self.SN_eps = SN_eps
        self.fp16 = G_fp16

        if self.resolution == 96:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 80:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 128:
            self.save_featrues = [0,1,2,3,4]
        elif self.resolution == 256:
            self.save_featrues = [0,1,2,3,4,5]
        elif self.resolution == 64:
            self.save_featrues = [0,1,2,3,4]
        self.out_channel_nultipiler = 1
        self.arch = content_encoder_arch(self.ch, self.out_channel_nultipiler,input_nc)[resolution]

        if self.G_param == 'SN':
            self.which_conv = functools.partial(SNConv2d,
                                                    kernel_size=3, padding=1,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
            self.which_linear = functools.partial(SNLinear,
                                                    num_svs=num_G_SVs, num_itrs=num_G_SV_itrs,
                                                    eps=self.SN_eps)
        self.blocks = []
        for index in range(len(self.arch['out_channels'][:-1])):

            self.blocks += [[DBlock(in_channels=self.arch['in_channels'][index],
                                             out_channels=self.arch['out_channels'][index],
                                             which_conv=self.which_conv,
                                             wide=self.G_wide,
                                             activation=self.activation,
                                             preactivation=(index > 0),
                                             downsample=nn.AvgPool2d(2))]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])
        self.init_weights()


    def init_weights(self):
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print('Param count for D''s initialized parameters: %d' % self.param_count)

    def forward(self,x):
        h = x
        residual_features = []
        residual_features.append(h)
        for index, blocklist in enumerate(self.blocks):
            for block in blocklist:
                h = block(h)            
            if index in self.save_featrues[:-1]:
                residual_features.append(h)     
        residual_features.append(h)        
        return h,residual_features

class UnifontModule(torch.nn.Module):
    '''https://github.com/aimagelab/VATr/blob/7952b16e4549811c442fb46ed49ac2585908e832/models/unifont_module.py#L6'''
    def __init__(self, 
                 alphabet,
                 out_dim,
                 device='cuda', 
                 input_type='unifont',
                 linear=True):
        super(UnifontModule, self).__init__()
        self.device = device
        self.alphabet = alphabet
        self.symbols = self.get_symbols('unifont')
        self.symbols_repr = self.get_symbols(input_type)

        if linear:
            self.linear = torch.nn.Linear(self.symbols_repr.shape[1], out_dim)
        else:
            self.linear = torch.nn.Identity()

    def get_symbols(self, input_type):
        with open(f"configs/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32).flatten() for sym in symbols}
        # self.special_symbols = [self.symbols[ord(char)] for char in special_alphabet]
        symbols = [symbols[ord(char)] for char in self.alphabet]
        symbols.insert(0, np.zeros_like(symbols[0]))
        symbols = np.stack(symbols)
        return torch.from_numpy(symbols).float().to(self.device)

    def forward(self, QR):
        return self.linear(self.symbols_repr[QR])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ContentEncoderV2(nn.Module):
    """
    Transformer block for image-like data. First, project the input (aka embedding) and reshape to b, t, d. Then apply
    standard transformer action. Finally, reshape to image.

    Parameters:
        in_channels (:obj:`int`): The number of channels in the input and output.
        n_heads (:obj:`int`): The number of heads to use for multi-head attention.
        d_head (:obj:`int`): The number of channels in each head.
        depth (:obj:`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (:obj:`float`, *optional*, defaults to 0.1): The dropout probability to use.
        context_dim (:obj:`int`, *optional*): The number of context dimensions to use.
    """

    def __init__(
        self,
        alphabet: str,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.pos_encoder = PositionalEncoding(in_channels, dropout)

        self.unifont_module = UnifontModule(alphabet, in_channels)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        hidden_states = self.unifont_module(hidden_states)
        # hidden_states = self.pos_encoder(hidden_states)
        residual = hidden_states
        # here change the shape torch.Size([1, 4096, 128])
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, context=context)  # hidden_states: torch.Size([1, 4096, 128])
        hidden_states = self.proj_out(hidden_states)
        return hidden_states + residual