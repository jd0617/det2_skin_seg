import torch
import torch.nn as nn
import torch.nn.functional as nnf

# from timm.models.layers import DropPath

class SE(nn.Module):
    def __init__(self, dim_in, reduction=16):
        super().__init__()
        self.dim_in = dim_in
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_in//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in//reduction, dim_in, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x*y


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7, bn=True, relu=False):
        super(SpatialGate, self).__init__()

        self.kernel_size = kernel_size
        self.bn = bn
        self.relu = relu

        self.compress = ChannelPool()
        self.spatial = self._make_spatial()

    def _make_spatial(self):
            
        layer = [nn.Conv2d(2, 1, kernel_size=self.kernel_size, 
                           stride=1, padding=(self.kernel_size-1)//2)
                ]
        if self.bn:
            layer.append(nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))
        if self.relu:
            layer.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layer)
        
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = nnf.sigmoid(x_out) # broadcasting
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.gap = nn.AdaptiveAvgPool2d((1))
        self.gmp = nn.AdaptiveMaxPool2d((1))
    def forward(self, x):
        avg_x = self.gap(x)
        max_x = self.gmp(x)

        avg_x = self.mlp(avg_x)
        max_x = self.mlp(max_x)

        scale = avg_x + max_x

        scale = nnf.sigmoid(scale).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
    
class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn = nn.Linear(dim, kernel_size**4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W

        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
            self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


# class Outlooker(nn.Module):
#     """
#     Implementation of outlooker layer: which includes outlook attention + MLP
#     Outlooker is the first stage in our VOLO
#     --dim: hidden dim
#     --num_heads: number of heads
#     --mlp_ratio: mlp ratio
#     --kernel_size: kernel size in each window for outlook attention
#     return: outlooker layer
#     """
#     def __init__(self, dim, kernel_size, padding, stride=1,
#                  num_heads=1,mlp_ratio=3., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm, qkv_bias=False,
#                  qk_scale=None):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
#                                      padding=padding, stride=stride,
#                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                      attn_drop=attn_drop)

#         self.drop_path = DropPath(
#             drop_path) if drop_path > 0. else nn.Identity()

#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim,
#                        hidden_features=mlp_hidden_dim,
#                        act_layer=act_layer)

#     def forward(self, x):
#         x = x + self.drop_path(self.attn(self.norm1(x)))
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x


class TBSFF(nn.Module):
    def __init__(self, in_dim, dim_red=8): #, num_branch=3 , drop_out=0.2
        super().__init__()
        self.in_dim = in_dim
        self.dim_red = dim_red
        hid_dim = in_dim // dim_red
        # self.num_branch = num_branch

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1,1))

        self.conv1 = nn.Conv2d(in_dim, hid_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # self.dropout = nn.Dropout(drop_out) if drop_out > 0.0 else nn.Indentity()

        self.tconv = nn.ConvTranspose2d(hid_dim, in_dim, kernel_size=(1, 3), bias=False)
        self.sm = nn.Softmax(-1)

    def forward(self, x:list[torch.Tensor]):

        xs = torch.stack(x).sum(0)

        xs = self.gap(xs) + self.gmp(xs)

        xs = self.conv1(xs)
        xs = self.relu(xs)

        xs = self.tconv(xs)
        xs = self.sm(xs)

        x1 = xs[:, :, :, 0].unsqueeze(-1) * x[0]
        x2 = xs[:, :, :, 1].unsqueeze(-1) * x[1]
        x3 = xs[:, :, :, 2].unsqueeze(-1) * x[2]

        return x1+x2+x3

            
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()

        self.dw_conv = nn.Conv2d(in_c, in_c, kernel_size=kernel_size,
                                 stride=stride, padding=padding,
                                 groups=in_c, bias=bias)
        
        self.pw_conv = nn.Conv2d(in_c, out_c, kernel_size=1,
                                 stride=1, padding=0, bias=bias)
        
    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)

        return x
