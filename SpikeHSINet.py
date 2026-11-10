"""
SpikeHSINet: a fully spiking spectral-spatial network for hyperspectral image classification.

Drop-in replacement for MS2GCAN:  input  [N, l*l, S]  ->  output logits [N, num_cls]

Design (all built from spikingjelly multi-step layers):
  1. Spectral-spatial stem      : 1x1 conv (S->C) -> BN -> LIF -> 3x3 conv -> BN   (direct encoding, repeated T times)
  2. L x SpikeBlock (membrane-shortcut, MS-ResNet style, LIF *before* every conv):
        a) MultiScaleConvMix    : gated sum of 1x1 / 3x3-DW / 5x5-DW branches   (local spatial context)
        b) SpikeSelfAttention   : Spikformer-style softmax-free spike Q,K,V     (global patch context)
        c) SpikeMLP             : 1x1 -> LIF -> 1x1                              (channel mixing)
  3. Readout                    : concat(center pixel, spatial mean) -> mean over T -> LayerNorm -> Linear
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


def lif(v_threshold: float = 1.0):
    return neuron.LIFNode(tau=2.0, v_threshold=v_threshold, detach_reset=True,
                          surrogate_function=surrogate.ATan())


def conv_bn(cin, cout, k=1, groups=1):
    return nn.Sequential(
        layer.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
        layer.BatchNorm2d(cout),
    )


class MultiScaleConvMix(nn.Module):
    """Learnable (softmax) mix of 1x1, 3x3-DW and 5x5-DW branches. Input/Output [T,N,C,H,W]."""
    def __init__(self, dim):
        super().__init__()
        self.lif = lif()
        self.b1 = conv_bn(dim, dim, 1)
        self.b3 = nn.Sequential(layer.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                                layer.Conv2d(dim, dim, 1, bias=False), layer.BatchNorm2d(dim))
        self.b5 = nn.Sequential(layer.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),
                                layer.Conv2d(dim, dim, 1, bias=False), layer.BatchNorm2d(dim))
        self.gate = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        s = self.lif(x)
        w = torch.softmax(self.gate, 0)
        return w[0] * self.b1(s) + w[1] * self.b3(s) + w[2] * self.b5(s)


class SpikeSelfAttention(nn.Module):
    """Spikformer-style SSA over the l*l pixels of the patch (no softmax, spikes only)."""
    def __init__(self, dim, heads=4, scale=0.125):
        super().__init__()
        assert dim % heads == 0, "hidden must be divisible by heads"
        self.h, self.scale = heads, scale
        self.lif_in = lif()
        self.q, self.k, self.v = conv_bn(dim, dim), conv_bn(dim, dim), conv_bn(dim, dim)
        self.q_lif, self.k_lif, self.v_lif = lif(), lif(), lif()
        self.attn_lif = lif(0.5)
        self.proj = conv_bn(dim, dim)

    def forward(self, x):
        T, N, C, H, W = x.shape
        s = self.lif_in(x)
        q = self.q_lif(self.q(s))
        k = self.k_lif(self.k(s))
        v = self.v_lif(self.v(s))

        def split(t):  # [T,N,C,H,W] -> [T,N,h,L,dh]
            return t.reshape(T, N, self.h, C // self.h, H * W).transpose(-1, -2)

        q, k, v = split(q), split(k), split(v)
        attn = (q @ k.transpose(-1, -2)) * self.scale        # [T,N,h,L,L]
        out = (attn @ v).transpose(-1, -2).reshape(T, N, C, H, W)
        out = self.attn_lif(out)
        return self.proj(out)


class SpikeMLP(nn.Module):
    def __init__(self, dim, ratio=2):
        super().__init__()
        self.lif1 = lif()
        self.fc1 = conv_bn(dim, dim * ratio)
        self.lif2 = lif()
        self.fc2 = conv_bn(dim * ratio, dim)

    def forward(self, x):
        return self.fc2(self.lif2(self.fc1(self.lif1(x))))


class SpikeBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.mix = MultiScaleConvMix(dim)
        self.attn = SpikeSelfAttention(dim, heads)
        self.mlp = SpikeMLP(dim)

    def forward(self, x):
        x = x + self.mix(x)
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SpikeHSINet(nn.Module):
    def __init__(self, T=4, img_size=15, num_cls=9, input_dim=103, hidden=64,
                 depth=2, heads=4, drop=0.2, use_cupy=False):
        super().__init__()
        self.T, self.l = T, img_size
        self.stem = nn.Sequential(
            layer.Conv2d(input_dim, hidden, 1, bias=False), layer.BatchNorm2d(hidden), lif(),
            layer.Conv2d(hidden, hidden, 3, padding=1, bias=False), layer.BatchNorm2d(hidden),
        )
        self.blocks = nn.ModuleList([SpikeBlock(hidden, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden * 2)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden * 2, num_cls)

        functional.set_step_mode(self, step_mode='m')
        if use_cupy:
            functional.set_backend(self, backend='cupy')

    def forward(self, x):                       # x: [N, l*l, S]
        functional.reset_net(self)
        N, L, S = x.shape
        l = self.l
        x = x.reshape(N, l, l, S).permute(0, 3, 1, 2)            # [N,S,l,l]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)            # [T,N,S,l,l]  (direct encoding)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        c = l // 2
        feat = torch.cat([x[..., c, c], x.mean(dim=(-1, -2))], dim=-1)   # [T,N,2C]
        feat = feat.mean(0)                                              # average over time -> [N,2C]
        return self.head(self.drop(self.norm(feat)))


def dihedral(x, l, k, flip):
    """Apply one of the 8 dihedral transforms to patches [N, l*l, S] (augmentation / TTA)."""
    N, L, S = x.shape
    x = x.reshape(N, l, l, S)
    if flip:
        x = torch.flip(x, dims=(2,))
    x = torch.rot90(x, k, dims=(1, 2))
    return x.reshape(N, L, S)
