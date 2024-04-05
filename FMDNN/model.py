
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import torch.nn.functional as F
from math import exp
import numpy as np
import torchvision.transforms as transform


def drop_path(x, drop_prob: float = 0., training: bool = False):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):


    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):


    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)  # 224*224
        patch_size = (patch_size, patch_size)  # 16*16
        self.img_size = img_size  # 224*224
        self.patch_size = patch_size  # 16*16
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 14*14
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 196

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # 768*14*14 ？
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # batch_size channel height width
        # flatten: [B, C, H, W] -> [B, C, HW]  [B196, 768, 14, 14] -> [B196, 768, 196]
        # transpose: [B, C, HW] -> [B, HW, C]   [B196, 768, 196] -> [B196, 196, 768]
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# #  ablation experiment 1
# class Attention(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
#                                                                                  3)  # B, num_heads, N, head_dim
#         k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
#                                                                                  3)  # B, num_heads, N, head_dim
#         v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
#                                                                                  3)  # B, num_heads, N, head_dim
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, N
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B, N, C
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


# #  ablation experiment 2
# class CrossAttention(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads=8,
#                  qkv_bias=False,
#                  qk_scale=None,
#                  attn_drop_ratio=0.,
#                  proj_drop_ratio=0.):
#         super(CrossAttention, self).__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.k = nn.Linear(dim, dim, bias=qkv_bias)
#         self.v = nn.Linear(dim, dim, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop_ratio)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop_ratio)
#
#     def forward(self, xpf):
#         # print(xpf.shape)
#         x = xpf[0]
#         x1 = xpf[1]
#
#         # print(x.shape)
#         # print(x1.shape)
#         # print('1111111111111111111')
#         # print(x.shape)
#         # print('22222222222')
#         B, N, C = x.shape
#         # print('333333333')
#         q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
#                                                                                   3)  # B, num_heads, N, head_dim
#         k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
#                                                                                  3)  # B, num_heads, N, head_dim
#         v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
#                                                                                  3)  # B, num_heads, N, head_dim
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale  # B, num_heads, N, N
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # B, N, C
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class FuzzyFeatureExtractor:
    def __init__(self, mu_params, sigma_params, trapezoidal_params, weights):
        self.mu_params = mu_params
        self.sigma_params = sigma_params
        self.trapezoidal_params = trapezoidal_params
        self.weights = weights

    def gaussian(self, x):
        mu, sigma = self.mu_params['mu'], self.mu_params['sigma']
        return lambda x: np.vectorize(lambda xi: norm.pdf(xi, mu, sigma))(x)

    def sigmoid(self, x):
        alpha, beta = self.sigma_params['alpha'], self.sigma_params['beta']
        return lambda x: np.vectorize(lambda xi: 1 / (1 + np.exp(-alpha * (xi - beta))))(x)

    def trapezoidal(self, x):
        a, b, c, d = self.trapezoidal_params['a'], self.trapezoidal_params['b'], self.trapezoidal_params['c'], \
                     self.trapezoidal_params['d']
        return lambda x: np.vectorize(lambda xi: np.piecewise(xi, [xi <= a, (a < xi) & (xi <= b), (b < xi) & (xi <= c),
                                                                   (c < xi) & (xi <= d), xi > d],
                                                              [0, lambda xi: (xi - a) / (b - a), 1,
                                                               lambda xi: (d - xi) / (d - c), 0]))(x)

    def fuzzy_feature_extraction(self, I):
        I_mu = self.gaussian(I)(I)
        I_sigma = self.sigmoid(I)(I)
        I_T = self.trapezoidal(I)(I)

        # Fuzzy feature matrix formation
        I_fuzzy = self.weights['w_mu'] * I_mu + self.weights['w_sigma'] * I_sigma + self.weights['w_T'] * I_T + \
                  self.weights['b']

        return I_fuzzy



class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()  # GELU
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block1(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block1, self).__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn = FCA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 768*4
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, xpf):
        x = self.norm1(xpf[0])
        xfuzzy = self.norm1(xpf[1])
        xpf = {0: x, 1: xfuzzy}
        # attention encoder
        x = x + self.drop_path(self.attn(xpf))  # norm1 + attention + dropout + x
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # norm2 + mlp + dropout + x
        # transformer encoder
        # x = x + self.drop_path(self.attn2(self.norm1(x)))  # norm1 + attention + dropout + x
        # x = x + self.drop_path(self.mlp(self.norm2(x)))  # norm2 + mlp + dropout + x
        # xpf = {0: x, 1: xfuzzy}

        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # norm1 + attention + dropout + x
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # norm2 + mlp + dropout + x

        return x



class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is [N,C,H,W]
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Transformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, zzn_channels=3, zzn_classes=3, bilinear=True):

        super(Transformer, self).__init__()

        self.n_channels = zzn_channels
        self.n_classes = zzn_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(zzn_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, zzn_classes)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1  # 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)  # partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU  # nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches


        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # 1 196+1 768
        self.pos_drop = nn.Dropout(p=drop_ratio)  # drop_ratio=0.

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay. rule drop_path_ratio=0.
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.Block1 = Block1(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale,
                             drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio,
                             norm_layer=norm_layer, act_layer=act_layer)
        # drop_path_ratio = dpr[i],
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)  # [B, 196, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward_features1(self, xpf):
        x = xpf[0]
        xfuzzy = xpf[1]
        # [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.patch_embed(x)
        xfuzzy = self.patch_embed(xfuzzy)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        fcls_token = self.cls_token.expand(xfuzzy.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # -> [B196, 197, 768]
            xfuzzy = torch.cat((fcls_token, xfuzzy), dim=1)  # -> [B196, 197, 768]
        else:  # 不看
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        xfuzzy = self.pos_drop(xfuzzy + self.pos_embed)

        xpf = {0: x, 1: xfuzzy}

        x = self.Block1(xpf)
        xpf = {0: x, 1: xfuzzy}
        x = self.Block1(xpf)
        xpf = {0: x, 1: xfuzzy}
        x = self.Block1(xpf)
        xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        # x = self.Block1(xpf)
        # xpf = {0: x, 1: xfuzzy}
        x = self.Block1(xpf)
        # x = self.blocks(x)

        # for 0 in range 12;
        #     x = self.Block(xpf)
        #     xpf = {0: x, 1: xfuzzy}

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        # print(x)

        xorg =
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        # print(x)  # 8,128,56,56
        xs = x
        conv_layer = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=2, padding=1).to('cuda')
        output_feature = conv_layer(xs)
        xs = F.interpolate(output_feature, size=(224, 224), mode='bilinear', align_corners=True)
        xs =  xs + xorg
        x = self.up3(x, x2)
        # print(x)  # 8,64,112,112
        xm = x
        conv_layer = nn.Conv2d(64, 3, kernel_size=3, stride=2, padding=1).to('cuda')
        output_tensor = conv_layer(xm)
        xm = F.interpolate(output_tensor, size=(224, 224), mode='bilinear', align_corners=False)
        xm = xm  + xorg
        x = self.up4(x, x1)
        # print(x)  # 8,64,224,224
        xl = self.outc(x)  + xorg  # 224*224*64  8,64,224,224
        xorg = self.forward_features(xorg)
        # print(xl)
        # x = self.patch_embed(x)
        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)
        # x = self.pos_drop(x + self.pos_embed)

        blur_transform = transform.GaussianBlur(kernel_size=3)
        xfuzzy = blur_transform(xfuzzy)


        xpf = {0: xs, 1: xfuzzy}
        xss = self.forward_features1(xpf)
        xpf = {0: xm, 1: xfuzzy}
        xmm = self.forward_features1(xpf)
        xpf = {0: xl, 1: xfuzzy}
        xll = self.forward_features1(xpf)
        x = xss + xmm +  xll

        if self.head_dist is not None:  #
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


def _init_vit_weights(m):

    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)




def vit_base_patch16_224_in21k(num_classes: int = 21843, has_logits: bool = True):

    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes)
    return model




'''
if __name__ == '__main__':
    model = vit_base_patch16_224_in21k()
    x = torch.rand(4, 3, 224, 224)
    out = model(x)
    print(out)
'''
