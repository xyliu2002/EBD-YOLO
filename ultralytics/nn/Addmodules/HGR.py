import torch
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class GnConv(nn.Module):
    def __init__(self, dim, order=2, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order

        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = nn.Conv2d(dim, dim, 7, 1, 3, groups=dim)
        else:
            self.dwconv = gflayer

        self.high_pass = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)

        kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32)

        kernel = kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)
        self.high_pass.weight = nn.Parameter(kernel, requires_grad=False)

        self.pws = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(order - 1)])

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.ln = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):

        fused_x = self.proj_in(x)
        p1, p2 = torch.chunk(fused_x, 2, dim=1)

        p2_high = self.high_pass(p2)

        p2 = p2 + 0.5 * p2_high

        for i, layer in enumerate(self.pws):
            p2 = self.dwconv(p2)

            p1 = p1 * p2

            p2 = layer(p2)

        p1 = p1.permute(0, 2, 3, 1)
        p1 = self.ln(p1)
        p1 = p1.permute(0, 3, 1, 2)

        return self.proj_out(p1)


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        super().__init__()

        self.attn = GnConv(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2HGR(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
