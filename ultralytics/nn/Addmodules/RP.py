import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):
    if d > 1:
        if isinstance(k, (tuple, list)):
            k = tuple(d * (x - 1) + 1 for x in k)
        else:
            k = d * (k - 1) + 1
    if p is None:
        if isinstance(k, (tuple, list)):

            def get_num(elem):
                if isinstance(elem, (tuple, list)):
                    return get_num(elem[0])
                return int(elem // 2)

            p = [get_num(x) for x in k]
        else:
            p = int(k // 2)
    return p


class SmallObjectAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())

    def forward(self, x):

        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_spatial, max_spatial], dim=1))

        return x * spatial_att


class RefineBlock(nn.Module):  # s=1
    def __init__(self, c_in, c_out, s=1, use_attention=True):
        super().__init__()
        self.s = s
        self.use_attention = use_attention

        self.branch_3x3 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, s, 1, bias=False), nn.BatchNorm2d(c_out))

        self.branch_1x1 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, s, 0, bias=False), nn.BatchNorm2d(c_out))

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, s, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c_out)
        )

        if s == 1 and c_in == c_out:
            self.branch_id = nn.BatchNorm2d(c_out)
        elif s == 2 and c_in == c_out:
            self.branch_id = nn.Sequential(nn.AvgPool2d(2, 2), nn.BatchNorm2d(c_out))
        else:
            self.branch_id = None

        num_branches = 3 if self.branch_id is None else 4
        self.branch_weights = nn.Parameter(torch.ones(num_branches) / num_branches)

        if use_attention:
            self.attention = SmallObjectAttention(c_out, reduction=8)

        self.act = nn.SiLU()

    def forward(self, x):

        branches = [self.branch_3x3(x), self.branch_1x1(x), self.branch_dilated(x)]

        if self.branch_id is not None:
            branches.append(self.branch_id(x))

        weights = F.softmax(self.branch_weights, dim=0)
        out = sum(w * b for w, b in zip(weights, branches))

        if self.use_attention:
            out = self.attention(out)

        return self.act(out)


class PreserveBlock(nn.Module):  # s=2
    def __init__(self, c_in, c_out, s=2):
        super().__init__()

        self.main = nn.Sequential(nn.Conv2d(c_in, c_out, 3, s, 1, bias=False), nn.BatchNorm2d(c_out), nn.SiLU())

        self.aux = nn.Sequential(nn.MaxPool2d(s, s), nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False), nn.BatchNorm2d(c_out))

        self.fuse = nn.Conv2d(c_out * 2, c_out, 1, 1, 0, bias=False)

    def forward(self, x):
        main_out = self.main(x)
        aux_out = self.aux(x)
        return self.fuse(torch.cat([main_out, aux_out], dim=1))


class RPConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, use_attention=True):
        super().__init__()
        if s == 2:
            self.unit = PreserveBlock(c1, c2, s)
        else:
            self.unit = RefineBlock(c1, c2, s=s, use_attention=use_attention)

    def forward(self, x):
        return self.unit(x)


class RefineBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k=3, e=1):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act1 = nn.SiLU()

        self.cv2 = RefineBlock(c_, c2, s=1, use_attention=True)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.act1(self.bn1(self.cv1(x))))
        return x + y if self.add else y


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        c1, c2, s, g, d = int(c1), int(c2), int(s), int(g), int(d)

        if isinstance(k, (list, tuple)):
            k = tuple(int(x) for x in k)
        else:
            k = int(k)

        pad = autopad(k, p, d)
        if isinstance(pad, (list, tuple)):
            pad = tuple(int(x) for x in pad)
        else:
            pad = int(pad)

        self.conv = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):

        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):

        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):

        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)

        self.cv2 = Conv(int((2 + n) * self.c), c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):

        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):

        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class RPC3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):

        super().__init__(c1, c2, n, shortcut, e, g)
        self.m = nn.ModuleList(RefineBlock(self.c, self.c, s=1) for _ in range(n))
