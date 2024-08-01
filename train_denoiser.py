import numpy as np
import optim
import torch
import os
from jinja2.compiler import F
from timm import DnCNN
from timm.layers import to_2tuple, DropPath, Mlp
from timm.models.swin_transformer import WindowAttention, window_partition, window_reverse
from torch import nn
from torch.library import _
from torch.utils.data import Dataset, DataLoader
import cv2

class SwinTransformer:
    pass
class ResidualBlock(nn.Module):
    def __init__(self,in_channles,num_channles,use_1x1conv=False,strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channles,num_channles,kernel_size=3,stride=strides,padding=1,)
        self.conv2 = nn.Conv2d(
            num_channles, num_channles, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(
                in_channles,num_channles,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channles)
        self.bn2=nn.BatchNorm2d(num_channles)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        y= F.relu(self.bn1(self.conv1(x)))
        y=self.bn2(self.conv2(y))
        if self.conv3:
            x=self.conv3(x)
        y+=x
        return F.relu(y)
blk=[]
blk.append(ResidualBlock(64, 64,use_1x1conv=False, strides=1))
blk.append(ResidualBlock(64, 128,use_1x1conv=True, strides=2))
print(blk)


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # DropPath（）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # 线性层

        # window
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1


            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # 把window展开成一行
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  #
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA，mask是否为None用以区分采用W-MSA还是SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class HybridFormer(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers=17, scales=[2, 3, 4]):
        super(HybridFormer, self).__init__()

        num_channels = 64

        # Define the model's parameters
        self.scales = scales
        self.residual_blocks = nn.ModuleList()
        self.swin_transformers = nn.ModuleList()
        self.convs = nn.ModuleList()

        # Use a different multi-scale architecture
        for scale in scales:
            self.residual_blocks.append(nn.Sequential(
                *[ResidualBlock(in_channels, num_channels) for _ in range(num_conv_layers)]
            ))
            self.swin_transformers.append(SwinTransformer(self, img_size=224 // scale, patch_size=4 // scale, window_size=8 // scale,
                  embed_dim=96 // scale, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                  mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0.2, norm_layer=nn.LayerNorm, ape=False, patch_norm=True))
            super().__init__()

        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        outputs = []
        for scale, residual_blocks, swin_transformer, conv in zip(self.scales, self.residual_blocks,
                                                                  self.swin_transformers, self.convs):
            identity = x

            # Residual blocks
            x = residual_blocks(x)

            # Swin Transformer module
            x = swin_transformer(x)

            # DeepCNN layers
            x = conv(x)

            # Add residual connection
            x += identity

            outputs.append(x)

        return outputs


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, num_channles, use_1x1conv=False, strides=1, in_channles=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, num_channles, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, num_channles, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channles, num_channles, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

    class HybridFormer_DnCNN(nn.Module):
        def __init__(self, upscale, img_size, window_size, img_range, depths, embed_dim, num_heads, mlp_ratio,
                     upsampler):
            super().__init__()
            self.HybridFormer = HybridFormer(upscale=upscale, img_size=img_size, window_size=window_size,
                                             img_range=img_range, depths=depths, embed_dim=embed_dim,
                                             num_heads=num_heads,
                                             mlp_ratio=mlp_ratio, upsampler=upsampler)
            self.dncnn = DnCNN(in_channels=3, out_channels=3)

    def forward(self, x):
        x = self.HybridFormer(x)
        x = self.dncnn(x)
        return x