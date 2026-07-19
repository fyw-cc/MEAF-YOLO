
"""Block modules required by MEAF-YOLO."""

import math
import typing as t

import torch
import torch.nn as nn
from einops import rearrange

from .conv import Conv

__all__ = (
    "Bottleneck",
    "C2f",
    "MRSE",
    "SDM",
    "FSDE",
    "CDGR",
    "TDCE",
    "TDC_C2f",
    "PLKF",
    "TD",
    "BU",
)


class MRSE(nn.Module):
    """Multi-Receptive Structural Encoder in the TDC-C2f module."""

    def __init__(self, c):
        super().__init__()
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dw7 = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.pw = Conv(3 * c, c, 1, 1)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.pw(torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))
        return self.act(self.bn(y)) + x


class SDM(nn.Module):
    """Spatial-Dependency Modulator in the TDC-C2f module."""

    def __init__(self, c):
        super().__init__()
        self.dw_5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dw_7_dil = nn.Conv2d(c, c, 7, 1, 9, dilation=3, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, bias=True)

    def forward(self, x):
        attn = self.pw(self.dw_7_dil(self.dw_5(x))).sigmoid()
        return x * attn


class FSDE(nn.Module):
    """Frequency-Selective Detail Enhancer in the TDC-C2f module."""

    def __init__(self, c, se_ratio=4):
        super().__init__()
        self.lowpass = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.lap = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, 1, bias=False)

        mid = max(c // se_ratio, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, mid, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, 1),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

        with torch.no_grad():
            self.lowpass.weight.copy_(torch.ones(c, 1, 5, 5) / 25.0)
            lap_k = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
            self.lap.weight.copy_(lap_k.view(1, 1, 3, 3).repeat(c, 1, 1, 1))

    def forward(self, x):
        x_h = x - self.lowpass(x)
        y = self.pw(self.dw3(self.lap(x_h)))
        return self.act(self.bn(y * self.se(y)))


class CDGR(nn.Module):
    """Cross-Domain Gated Refiner used after TDCE aggregation."""

    def __init__(self, c, se_ratio=4):
        super().__init__()
        self.spatial = nn.Conv2d(c, 1, 1, 1)
        self.dw = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        mid = max(c // se_ratio, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, mid, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, 1),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = x * torch.sigmoid(self.spatial(x))
        x = self.dw(x)
        return self.act(self.bn(x * self.se(x)))


class TDCE(nn.Module):
    """Tri-Domain Collaborative Enhancer composed of MRSE, SDM, and FSDE."""

    def __init__(self, c, se_ratio=4):
        super().__init__()
        # Keep attribute names stable so existing state dictionaries remain loadable.
        self.incep = MRSE(c)
        self.lka = SDM(c)
        self.wave = FSDE(c, se_ratio=se_ratio)
        self.fuse = nn.Conv2d(3 * c, c, 1, 1, bias=False)

        mid = max(c // se_ratio, 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, mid, 1, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, 1),
            nn.Sigmoid(),
        )
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        f = self.fuse(torch.cat([self.incep(x), self.lka(x), self.wave(x)], dim=1))
        return self.act(self.bn(f * self.se(f))) + x


class TDC_C2f(nn.Module):
    """Tri-Domain Collaborative C2f module (TDC-C2f in the paper)."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.m = nn.ModuleList(TDCE(self.c_) for _ in range(n))
        self.cv2 = Conv((2 + n) * self.c_, c2, 1, 1)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.gate = CDGR(c2)
        self.alpha = nn.Parameter(torch.tensor(0.15))
        self.beta = nn.Parameter(torch.tensor(0.40))
        self.use_sc = shortcut

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        out = self.cv2(torch.cat(y, 1))
        skip = x if isinstance(self.proj, nn.Identity) else self.proj(x)
        fused = out + self.alpha * (skip if self.use_sc or not isinstance(self.proj, nn.Identity) else 0)
        return fused + self.beta * self.gate(fused)


class DWLargeKernel(nn.Module):
    def __init__(self, c, k=7, d=1, act=True):
        super().__init__()
        self.dw1 = Conv(c, c, (k, 1), 1, d=d, g=c, act=act)
        self.dw2 = Conv(c, c, (1, k), 1, d=d, g=c, act=act)

    def forward(self, x):
        return self.dw2(self.dw1(x))


class DPLKA_Module(nn.Module):
    def __init__(self, c, large_k=9, d=1, reduction=4, use_pw=True, act=True):
        super().__init__()
        self.dw = DWLargeKernel(c, k=large_k, d=d, act=act)
        self.pw = Conv(c, c, 1, 1, act=act) if use_pw else nn.Identity()
        hidden = max(1, c // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(inplace=True) if act else nn.Identity(),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.pw(self.dw(x))
        return out * self.gate(out)


class Branch(nn.Module):
    def __init__(self, dim, act=True):
        super().__init__()
        self.one = dim // 2
        self.two = dim - self.one
        self.dw = Conv(self.one, self.one, 3, 1, g=self.one, act=act)
        self.pw = Conv(self.one, dim, 1, 1, act=act)
        self.conv2 = Conv(self.two, dim, 1, 1, act=act)
        self.spatial_conv = nn.Conv2d(dim, 1, 1, 1)
        self.spatial_bn = nn.BatchNorm2d(1)
        self.spatial_sigmoid = nn.Sigmoid()
        self.channel_dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        x3 = self.pw(self.dw(x1))
        x4 = self.conv2(x2)
        spatial_att = self.spatial_sigmoid(self.spatial_bn(self.spatial_conv(x4)))
        channel_att = self.channel_sigmoid(self.channel_pool(self.channel_dwconv(x3)))
        return spatial_att * x3 + channel_att * x4


class PLKF(nn.Module):
    def __init__(self, c1, c2, k=5, act=True):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lka_list = nn.ModuleList(
            [
                DPLKA_Module(c_, large_k=11, d=2, reduction=8, use_pw=True, act=act),
                DPLKA_Module(c_, large_k=9, d=2, reduction=8, use_pw=True, act=act),
                DPLKA_Module(c_, large_k=7, d=1, reduction=8, use_pw=True, act=act),
                DPLKA_Module(c_, large_k=5, d=1, reduction=8, use_pw=True, act=act),
            ]
        )
        self.batch = Branch(c_, act=act)

    def forward(self, x):
        y0 = self.cv1(x)
        y = [y0]
        y.extend(self.m(y[-1]) for _ in range(3))
        x_ = None
        for idx, m_ in enumerate(self.lka_list):
            x_ = m_(y[idx]) if idx == 0 else m_(x_ + y[idx])
        return self.cv2(torch.cat([y0, y[-1], x_, self.batch(y0)], 1))


class Bottleneck(nn.Module):
    """Standard bottleneck used by C2f."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class channel_att(nn.Module):
    def __init__(self, channel: int, b: int = 1, gamma: int = 2):
        super().__init__()
        k = int(abs((math.log(channel, 2) + b) / gamma))
        k = k if (k % 2 == 1) else (k + 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)


class spatial_att(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str, channelAttention_reduce: int = 4):
        super().__init__()
        assert in_channels == out_channels
        assert mode in ("p2", "p3", "p4")
        c = in_channels
        if mode == "p2":
            self.sa = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
        elif mode == "p3":
            self.sa = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
        else:
            self.sa = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=(1, 7), padding=(0, 3), groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=(7, 1), padding=(3, 0), groups=c, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )
        self.gate = nn.Sequential(nn.Conv2d(c, c, kernel_size=1, bias=False), nn.BatchNorm2d(c), nn.Sigmoid())

    def forward(self, x: torch.Tensor, caf: torch.Tensor = None) -> torch.Tensor:
        return caf * self.gate(self.sa(x))


class TD(nn.Module):
    """Top-down feature-fusion module of CD-PAN (``td`` in Figure 1)."""

    def __init__(self, c1: int, c2: t.Union[int, t.Sequence[int]], mode: str, n: int = 1):
        super().__init__()
        assert mode in ("p2", "p3", "p4")
        self.ca = channel_att(c1)
        self.sa = spatial_att(c1, c1, mode)
        if isinstance(c2, int):
            self.align = Conv(c2, c1, 1)
            self.possese = None
        else:
            c2 = list(c2)
            self.possese = nn.ModuleList()
            for i in range(len(c2) - 1):
                k = 7 if i == 0 else 5
                p = 3 if i == 0 else 2
                self.possese.append(
                    nn.Sequential(
                        nn.MaxPool2d(k, 2, p),
                        nn.Conv2d(c2[i], c2[i + 1], 1, bias=False),
                        nn.BatchNorm2d(c2[i + 1]),
                        nn.ReLU(inplace=True),
                    )
                )
            self.align = Conv(c2[-1], c1, 1)

    def _merge_low(self, xs: t.Sequence[torch.Tensor]) -> torch.Tensor:
        if self.possese is None:
            return xs[0]
        out = xs[0]
        for i, block in enumerate(self.possese):
            out = block(out) + xs[i + 1]
        return out

    def forward(self, x: t.Sequence[torch.Tensor]) -> torch.Tensor:
        x1 = self.align(self._merge_low(x[1:]))
        fuse = x[0] + x1
        return self.sa(x1, self.ca(x[0]) * fuse)


class BU(nn.Module):
    """Bottom-up feature-fusion module of CD-PAN (``bu`` in Figure 1)."""

    def __init__(
        self,
        dim: int,
        head_num: int,
        window_size: int = 7,
        down_sample_mode: str = "max_pool",
        attn_drop_ratio: float = 0.5,
        high_c: t.Optional[t.Sequence[int]] = None,
        group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
        qkv_bias: bool = False,
        fuse_bn: bool = False,
        norm_cfg: t.Dict = dict(type="BN"),
        act_cfg: t.Dict = dict(type="ReLU"),
        gate_layer: str = "sigmoid",
    ):
        super().__init__()
        assert dim % 4 == 0, f"dim must be divisible by 4, got {dim}"
        assert dim % head_num == 0, f"dim must be divisible by head_num, got dim={dim}, head_num={head_num}"
        assert gate_layer in ("sigmoid", "softmax")
        assert down_sample_mode in ("recombination", "avg_pool", "max_pool")

        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim**-0.5
        self.group_chans = dim // 4
        self.high_c = list(high_c) if high_c is not None else None
        if self.high_c is not None and len(self.high_c) == 2:
            self.poseese = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(self.high_c[0], self.high_c[1], 1, bias=False),
                nn.BatchNorm2d(self.high_c[1]),
                nn.SiLU(inplace=True),
            )
        else:
            self.poseese = None

        ks = group_kernel_sizes
        self.local_dwc = nn.Conv1d(self.group_chans, self.group_chans, ks[0], padding=ks[0] // 2, groups=self.group_chans)
        self.global_dwc_s = nn.Conv1d(self.group_chans, self.group_chans, ks[1], padding=ks[1] // 2, groups=self.group_chans)
        self.global_dwc_m = nn.Conv1d(self.group_chans, self.group_chans, ks[2], padding=ks[2] // 2, groups=self.group_chans)
        self.global_dwc_l = nn.Conv1d(self.group_chans, self.group_chans, ks[3], padding=ks[3] // 2, groups=self.group_chans)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == "softmax" else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)
        self.window_size = window_size
        self.down_sample_mode = down_sample_mode
        self.conv_d = nn.Identity()
        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        elif down_sample_mode == "recombination":
            self.down_func = self.space_to_chans
            self.conv_d = nn.Conv2d(dim * window_size**2, dim, 1, bias=False)
        elif down_sample_mode == "avg_pool":
            self.down_func = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        else:
            self.down_func = nn.MaxPool2d(kernel_size=window_size, stride=window_size)

        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(dim, dim, 1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == "softmax" else nn.Sigmoid()

    @staticmethod
    def space_to_chans(x: torch.Tensor, window_size: int = 7) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = window_size
        assert h % ws == 0 and w % ws == 0
        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        return x.view(b, c * ws * ws, h // ws, w // ws)

    def _spatial_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(dim=3)
        x_w = x.mean(dim=2)
        l_x_h, g1_h, g2_h, g3_h = torch.split(x_h, self.group_chans, dim=1)
        l_x_w, g1_w, g2_w, g3_w = torch.split(x_w, self.group_chans, dim=1)
        h_cat = torch.cat(
            (self.local_dwc(l_x_h), self.global_dwc_s(g1_h), self.global_dwc_m(g2_h), self.global_dwc_l(g3_h)),
            dim=1,
        )
        w_cat = torch.cat(
            (self.local_dwc(l_x_w), self.global_dwc_s(g1_w), self.global_dwc_m(g2_w), self.global_dwc_l(g3_w)),
            dim=1,
        )
        x_h_attn = self.sa_gate(self.norm_h(h_cat)).view(b, c, h, 1)
        x_w_attn = self.sa_gate(self.norm_w(w_cat)).view(b, c, 1, w)
        return x_h_attn * x_w_attn

    def _channel_attn(self, y: torch.Tensor) -> torch.Tensor:
        y = self.down_func(y)
        if self.down_sample_mode == "recombination" and self.window_size != -1:
            y = self.space_to_chans(y, self.window_size)
        y = self.norm(self.conv_d(y))
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        _, _, h, w = q.shape
        q = rearrange(q, "b (hn hd) h w -> b hn hd (h w)", hn=self.head_num, hd=self.head_dim)
        k = rearrange(k, "b (hn hd) h w -> b hn hd (h w)", hn=self.head_num, hd=self.head_dim)
        v = rearrange(v, "b (hn hd) h w -> b hn hd (h w)", hn=self.head_num, hd=self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v
        out = rearrange(out, "b hn hd (h w) -> b (hn hd) h w", h=h, w=w)
        return self.ca_gate(out.mean((2, 3), keepdim=True))

    def forward(self, x_: t.Sequence[torch.Tensor]) -> torch.Tensor:
        if self.high_c is not None and self.poseese is not None:
            x1_aligned = self.poseese(x_[1]) + x_[2]
            fuse = x_[0] + x1_aligned
            y_for_ca = x1_aligned
            x_for_sa = x_[0]
        else:
            fuse = x_[0] + x_[1]
            y_for_ca = x_[1]
            x_for_sa = x_[0]
        return self._channel_attn(y_for_ca) * (fuse * self._spatial_attn(x_for_sa))
