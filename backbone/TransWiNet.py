import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class SpaceToDepthRaw1d(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.stride = stride

    def forward(self, x):
        # x shape: [B, C, L]
        even = x[..., ::2]  # [B, C//2, L]
        odd = x[..., 1::2]  # [B, C//2, L]

        # Stack and reshape to interleave: [B, C//2, L*2]
        interleaved = torch.stack([even, odd], dim=self.d + 1)  # [B, C//2, 2, L]
        shape = list(interleaved.shape)
        shape[self.d] *= shape[self.d + 1]
        shape.pop(self.d + 1)
        return interleaved.reshape(*shape)

class SpaceToDepthRaw(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        return F.pixel_unshuffle(x, downscale_factor=self.stride)


class DepthToSpaceRaw(nn.Module):
    def __init__(self, stride=2, dimension=1):
        super().__init__()
        self.stride = stride
        self.d = dimension  

    def forward(self, x):
        return F.pixel_shuffle(x, upscale_factor=self.stride)


class ConvBN1d(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True,
                 with_bias=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv1d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups,
                                                bias=with_bias))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm1d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True,
                 with_bias=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups,
                                                bias=with_bias))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        if stride != 1:
            ratio = 2
            self.std = SpaceToDepthRaw1d()
        else:
            self.std = nn.Identity()
            ratio = 1
        self.layer = ConvBN1d(in_channels * ratio, out_channels, 3, padding=1, stride=1, with_bn=True)

    def forward(self, x):
        return self.layer(self.std(x.transpose(1, 2))).transpose(1, 2)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, h_dim: int, stride: int = 1, num_heads=4, new_block=True) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(in_planes)
        if new_block:
            self.conv1 = [self.norm,
                          DownBlock(in_planes, planes, stride=stride),
                          self.act]
            self.conv1 = torch.nn.Sequential(*self.conv1)
            self.shortcut = ConvBN1d(in_planes, planes, 3, padding=(3 - 1) // 2, stride=stride, with_bn=False,
                                     groups=1)
            self.pos_embed = nn.Parameter(torch.zeros(1, 200, planes))  # or dynamic
            h_dim = h_dim // stride
        else:
            self.conv1 = nn.Identity()
            self.shortcut = nn.Identity()
            self.pos_embed = torch.zeros(1, 200, planes)
        self.layer = EnhancedTransformerBlock(h_dim=h_dim, dim=planes, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """

        out = self.conv1(x)
        out = self.layer(out + self.pos_embed[:, :out.size(1), :].to(device=out.device))
        out = out + self.shortcut(x.transpose(1, 2)).transpose(1, 2)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x) # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


from timm.models.layers import DropPath


class AttInvMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, h_dim, num_heads, layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.v_s_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(2 * embed_dim, embed_dim)

    def forward(self, q, k, v):  # q, k, v: [B, N, C]
        B, N, C = q.shape

        # Linear projections
        v_s = self.v_s_proj(v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = self.q_proj(q).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d//h]
        k = self.k_proj(k).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = F.relu(
            torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim))  # [B, h, N, d//h] @ [B, h, d//h, n] -> [B, h, N, N]

        attn_scores_space = F.relu(torch.matmul(q.transpose(-2, -1), k) / (
            k.size(-2)))  # [B, h, d//h, N] @ [B, h, N, d//h] -> [B, h, d//h, d//h]

        # Apply attention
        out = torch.matmul(attn_scores, v)  # [B, h, N, N] * [B, h, N, d//h] -> [B, h, N, d//h]
        out = out.transpose(1, 2).contiguous().view(B, N, C)  # [B, h, N, d//h] -> [B, N, h, d//h] -> [B, N, C]

        out_s = torch.matmul(attn_scores_space, v_s.transpose(-2, -1)).contiguous().view(B, C, N).transpose(-2,
                                                                                                            -1)  #  [B, h, d//h, d//h] * [B, h, N, d//h]-> [B, h, d//h, N]
        out = torch.concat((out, out_s), dim=-1)
        return self.out_proj(out), [attn_scores, attn_scores_space]  # [B, N, C]



class WindowAttention2d(nn.Module):
    def __init__(self, h_dim, w_dim, num_heads):
        """
        window_size: int or tuple[int, int], (Wh, Ww)
        num_heads: number of attention heads
        """
        super().__init__()
        self.window_area = w_dim
        self.attn = AttInvMultiheadAttention(embed_dim=self.window_area, h_dim=h_dim, num_heads=num_heads,
                                                batch_first=True)


        if h_dim <= w_dim:
            Wh, Ww = h_dim // 4, self.window_area // (h_dim // 4)
        else:
            Wh, Ww = self.window_area // (w_dim // 4), w_dim // 4
        self.Wh, self.Ww = Wh, Ww

        win_stride = 2
        self.space_conv = nn.Sequential(*[
            nn.Conv2d(h_dim // Wh * w_dim // Ww // win_stride ** 2, h_dim // Wh * w_dim // Ww // win_stride ** 2,
                      kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(h_dim // Wh * w_dim // Ww // win_stride ** 2, h_dim // Wh * w_dim // Ww // win_stride ** 2,
                      kernel_size=3, stride=1, padding=1),
        ])
        self.dts = DepthToSpaceRaw(stride=win_stride)
        self.std = SpaceToDepthRaw(stride=win_stride)

    def forward(self, input, *args):
        x = input
        B, T, D = x.shape

        Wh, Ww = self.Wh, self.Ww

        # Reshape x to [B, H, W], where H=T, W=D, so we treat x as 2D feature map
        x_2d = x  # [B, T, D]

        Hp, Wp = x_2d.shape[1], x_2d.shape[2]  # padded H and W

        # Now reshape to windows
        x_win = x_2d.view(B, Hp // Wh, Wh, Wp // Ww, Ww)  # [B, Nh, Wh, Nw, Ww]
        x_win = x_win.permute(0, 1, 3, 2, 4).contiguous()  # [B, Nh, Nw, Wh, Ww]
        x_win = x_win.view(B, Hp // Wh * Wp // Ww, Wh, Ww)  # [B * Nh * Nw, 1, window_area]
        x_win = self.dts(x_win)
        x_win = self.space_conv(x_win)
        x_win = self.std(x_win)
        x_win = x_win.view(B, Hp // Wh * Wp // Ww, Wh * Ww)  # [B, Nh, Nw, window_area]

        # Apply attention: use same x as q, k, v
        out_2d, _ = self.attn(input, x_win, x_win)

        return out_2d, _


class EnhancedTransformerBlock(nn.Module):
    def __init__(self, h_dim, dim, num_heads=4, mlp_ratio=4.0, dropout=0, drop_path=0):
        super().__init__()
        # self.win_attn = WindowAttention(dim=dim, num_heads=num_heads, window_size=4)
        self.win_attn = WindowAttention2d(num_heads=num_heads, w_dim=dim, h_dim=h_dim)

        self.attn1 = AttInvMultiheadAttention(embed_dim=dim, h_dim=h_dim, num_heads=num_heads, batch_first=True)


        self.k_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
        self.v_conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

        self.dropout = nn.Dropout(dropout)

        # self.use_local_conv = use_local_conv
        # if use_local_conv:
        self.local_conv1 = nn.Linear(dim, dim)
        self.local_conv2 = nn.Linear(dim, dim)
        self.local_conv3 = nn.Linear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.norm1(x)
        attn_out, _ = self.win_attn(x, x, x)
        x = res + self.drop_path(self.dropout(attn_out))

        res = x
        x = self.norm2(x)
        x = res + self.drop_path(self.dropout(self.mlp1(x)))

        res = x
        x = self.norm3(x)
        x_conv = x.transpose(1, 2)
        k = self.k_conv(x_conv).transpose(1, 2)
        v = self.v_conv(x_conv).transpose(1, 2)
        attn_out, _ = self.attn1(x, k, v)
        x = res + self.drop_path(self.dropout(attn_out))

        res = x
        x = self.norm4(x)
        x = res + self.drop_path(self.dropout(self.mlp2(x)))

        res = x
        x = self.norm5(x)
        x_local1, x_local2 = self.local_conv1(x), self.local_conv2(x)
        out1 = x_local1 * self.act(x_local2)
        out2 = x_local2 * self.act(x_local1)
        x_local = self.local_conv3(out1 + out2)
        x = res + self.act(x_local)

        return x


class Stem(nn.Module):
    def __init__(self, in_planes: int, planes: int, stem_std=False):
        super().__init__()

        if stem_std:
            self.std1d = SpaceToDepthRaw1d(dimension=1)
            ratio = 2
        else:
            self.std1d = nn.Identity()
            ratio = 1

        self.conv1 = ConvBN(3 * ratio, 12, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBN(12, 12, kernel_size=3, stride=1, padding=1)


        self.layer = nn.Linear(in_planes // ratio // 2, planes, bias=True)
        self.act = nn.GELU()

        self.residual = nn.Linear(in_planes // 2, planes, bias=True)
        self.h_dim = in_planes // 2
        # self.std2d = SpaceToDepthRaw()
        self.average_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # raw_x = self.std2d(x)
        raw_x = self.average_pool(x)
        x = self.std1d(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x).mean(dim=1)
        x = self.layer(x) + self.residual(raw_x.mean(dim=1))

        return x


class BackBone(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, name: str, input_shape=(128, 128), layer_dims=(64, 128, 256, 512),
                 strides=(2, 2, 2, 1),
                 num_heads=4, stem_std=False) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(BackBone, self).__init__()
        self.name = name
        self.in_planes = layer_dims[0]
        self.block = block
        self.num_classes = num_classes
        nf = layer_dims[0]
        self.nf = nf
        self.input_shape = input_shape

        self.stem = Stem(in_planes=input_shape[1], planes=nf, stem_std=stem_std)
        self.h_dim = self.stem.h_dim

        self.layers = []
        self.strides = strides

        for i in range(len(layer_dims)):
            stride = self.strides[i]
            self.layers += [self._make_layer(block, layer_dims[i], num_blocks[i], stride=stride, num_heads=num_heads)]
        self.layers = torch.nn.Sequential(*self.layers)

        self.cls = nn.Linear(layer_dims[-1], num_classes)

        self.out_features = layer_dims[-1]

        self.avgpool = nn.AdaptiveAvgPool1d((1,))


    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int, num_heads=4) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        new_block = True

        for stride in strides:
            layers.append(
                block(self.in_planes, planes, h_dim=self.h_dim, stride=stride, num_heads=num_heads,
                      new_block=new_block))
            self.in_planes = planes * block.expansion
            self.h_dim = self.h_dim // stride
            new_block = False
        return nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.layers(out)
        # feature = out[:, 0]
        out = self.avgpool(out.transpose(1, 2))  # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512
        return feature

    def classifier(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cls(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = self.cls(out)
        return out



class TransWi(nn.Module):
    def __init__(self, num_classes, input_shape=(128, 128), num_blocks=(2, 2, 2, 2), name='TransWiNet',
                 layer_dims=(64, 128, 256, 512),
                 strides=(2, 2, 2, 1),
                 num_heads=4, stem_std=False):
        super(TransWi, self).__init__()
        self.name = 'TransWiNet'

        self.num_classes = num_classes

        self.backbone = BackBone(BasicBlock, num_blocks, num_classes, name, input_shape=input_shape,
                                 layer_dims=layer_dims, strides=strides,
                                 num_heads=num_heads, stem_std=stem_std)
        self.out_features = self.backbone.out_features 
        self.cls = self.backbone.cls


    def forward_class(self, features):
        return self.cls(features)

    def forward(self, x, **kwargs):
        x = self.features(x)
        return self.forward_class(x)

    def features(self, x, **kwargs):
        raw_features = self.backbone.features(x)
        return raw_features


def TransWiNet_Small(num_classes, args=None):
    return TransWi(num_classes=num_classes, num_blocks=(1, 1, 1), name='TransWiNet-S', layer_dims=(32, 48, 64),
                    strides=(1, 2, 2, 1), num_heads=4, stem_std=True)  # (32, 48, 64)


def TransWiNet(num_classes, args=None):
    return TransWi(num_classes=num_classes, num_blocks=(1, 1, 1), name='TransWiNet', layer_dims=(64, 128, 256),
                    strides=(1, 2, 2, 1), num_heads=4)  # (48, 96, 192)


