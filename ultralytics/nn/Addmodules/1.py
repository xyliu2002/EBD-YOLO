import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# =============================================================================
# 小目标注意力模块
# =============================================================================
class SmallObjectAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 通道注意力
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        # 空间注意力 (保留高频细节，对小目标关键)
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False), nn.Sigmoid())

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        # 空间注意力
        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_spatial, max_spatial], dim=1))

        return x * spatial_att


# =============================================================================
# 改进版 RFE_Block_S (Small Object Enhanced)
# =============================================================================
class RFE_Block_S(nn.Module):
    def __init__(self, c_in, c_out, s=1, use_attention=True):
        super().__init__()
        self.s = s
        self.use_attention = use_attention

        # 分支 1: 3x3 主干 (局部特征)
        self.branch_3x3 = nn.Sequential(nn.Conv2d(c_in, c_out, 3, s, 1, bias=False), nn.BatchNorm2d(c_out))

        # 分支 2: 1x1 辅助 (通道信息 + 定位修正)
        self.branch_1x1 = nn.Sequential(nn.Conv2d(c_in, c_out, 1, s, 0, bias=False), nn.BatchNorm2d(c_out))

        # 分支 3: 3x3 空洞卷积 (扩大感受野，对小目标关键!)
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, s, padding=2, dilation=2, bias=False), nn.BatchNorm2d(c_out)
        )

        # 分支 4: Identity (特征保持)
        if s == 1 and c_in == c_out:
            self.branch_id = nn.BatchNorm2d(c_out)
        elif s == 2 and c_in == c_out:
            self.branch_id = nn.Sequential(nn.AvgPool2d(2, 2), nn.BatchNorm2d(c_out))
        else:
            self.branch_id = None

        # 分支权重学习 (自适应融合)
        num_branches = 3 if self.branch_id is None else 4
        self.branch_weights = nn.Parameter(torch.ones(num_branches) / num_branches)

        # 小目标注意力增强
        if use_attention:
            self.attention = SmallObjectAttention(c_out, reduction=8)

        self.act = nn.SiLU()

    def forward(self, x):
        # 收集各分支输出
        branches = [self.branch_3x3(x), self.branch_1x1(x), self.branch_dilated(x)]

        if self.branch_id is not None:
            branches.append(self.branch_id(x))

        # 自适应加权融合
        weights = F.softmax(self.branch_weights, dim=0)
        out = sum(w * b for w, b in zip(weights, branches))

        # 注意力增强
        if self.use_attention:
            out = self.attention(out)

        return self.act(out)


# =============================================================================
# 高分辨率特征保持模块 (HRFP)
# =============================================================================
class HighResFeaturePreserve(nn.Module):
    def __init__(self, c_in, c_out, s=2):
        super().__init__()

        # 主路径: 标准下采样
        self.main = nn.Sequential(nn.Conv2d(c_in, c_out, 3, s, 1, bias=False), nn.BatchNorm2d(c_out), nn.SiLU())

        # 辅助路径: 最大池化保留边缘 (小目标通常有较强边缘响应)
        self.aux = nn.Sequential(nn.MaxPool2d(s, s), nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False), nn.BatchNorm2d(c_out))

        # 融合
        self.fuse = nn.Conv2d(c_out * 2, c_out, 1, 1, 0, bias=False)

    def forward(self, x):
        main_out = self.main(x)
        aux_out = self.aux(x)
        return self.fuse(torch.cat([main_out, aux_out], dim=1))


# =============================================================================
# 原子算子: RFEConv_S
# =============================================================================
class XConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True, use_attention=True):
        super().__init__()
        if s == 2:
            # 下采样时使用高分辨率保持
            self.unit = HighResFeaturePreserve(c1, c2, s)
        else:
            self.unit = RFE_Block_S(c1, c2, s=s, use_attention=use_attention)

    def forward(self, x):
        return self.unit(x)


# =============================================================================
# 模块: RFE_Bottleneck_S
# =============================================================================
class RFE_Bottleneck_S(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k=3, e=1.0):
        super().__init__()
        c_ = int(c2 * e)

        # 1. 1x1 投影 (保持通道信息)
        self.cv1 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_)
        self.act1 = nn.SiLU()

        # 2. RFE_S 提取 (带注意力)
        self.cv2 = RFE_Block_S(c_, c2, s=1, use_attention=True)

        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.act1(self.bn1(self.cv1(x))))
        return x + y if self.add else y


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(RFE_Bottleneck_S(c_, c_) for _ in range(n)))


class XC3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else RFE_Bottleneck_S(self.c, self.c, False) for _ in range(n)
        )
