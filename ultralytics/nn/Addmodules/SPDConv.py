import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class LiteSPD(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        c1_spd = c1 * 4

        self.pw_conv = nn.Conv2d(c1_spd, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)

        self.dw_conv = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

        x = self.act(self.bn1(self.pw_conv(x)))
        x = self.act(self.bn2(self.dw_conv(x)))
        return x
