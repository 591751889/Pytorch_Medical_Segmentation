# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import math
import torch
import torch.nn as nn
from functools import partial

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
import numpy as np


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]


class DeiT(VisionTransformer):
    """
    改造点：
    - 不使用 cls_token（与原代码一致）。
    - forward 时根据输入尺寸(H,W) -> (H//16, W//16) 动态插值 pos_embed。
    - 需要在构造时设置 self.pos_hw 为初始化时 pos_embed 的二维网格尺寸（如 12x16 或 24x32）。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_patches = self.patch_embed.num_patches
        # 注意：这里先按 N+1 初始化（保持和上游结构兼容），后面工厂函数会裁掉 cls，并重置为 [1, N, D]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_hw = None  # (H0, W0) —— 原始二维网格，工厂函数里会设置

    def _infer_hw_from_N(self, N):
        # 偏向 3:4 的分解，其次退化为因子分解
        for k in range(1, 100):
            h, w = 3 * k, 4 * k
            if h * w == N:
                return (h, w)
        h = int(round(math.sqrt(N)))
        while N % h != 0:
            h -= 1
        return (h, N // h)

    def _resize_pos_embed(self, pe, target_hw):
        """
        pe: [1, N0, D]（不含 cls）
        target_hw: (Ht, Wt)
        """
        B0, N0, D = pe.shape
        if self.pos_hw is None:
            self.pos_hw = self._infer_hw_from_N(N0)
        H0, W0 = self.pos_hw
        pe2d = pe.transpose(1, 2).contiguous().view(1, D, H0, W0)   # [1, D, H0, W0]
        pe2d = F.interpolate(pe2d, size=target_hw, mode='bilinear', align_corners=True)
        pe_new = pe2d.flatten(2).transpose(1, 2).contiguous()       # [1, Ht*Wt, D]
        return pe_new

    def forward(self, x):
        # 与原版一致：无 cls_token，仅 patch tokens
        B, C, H, W = x.shape
        x = self.patch_embed(x)                          # [B, N, D]
        Ht, Wt = H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]
        N = Ht * Wt

        pe = self.pos_embed
        # 如果 pos_embed 还有 cls_token（N+1），裁掉第一个
        if pe.shape[1] == self.patch_embed.num_patches + 1:
            pe = pe[:, 1:, :]

        if pe.shape[1] != N:
            pe = self._resize_pos_embed(pe, (Ht, Wt))

        x = x + pe
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=384, depth=8, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_small_patch16_224-cd65a155.pth', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)

    # 初始化时：去掉 cls_token，并插值到 (12,16)，记录 pos_hw
    pe = model.pos_embed[:, 1:, :].detach()                 # [1, N, D]
    pe = pe.transpose(-1, -2)                               # [1, D, N]
    side = int(np.sqrt(pe.shape[2]))
    pe = pe.view(pe.shape[0], pe.shape[1], side, side)      # [1, D, S, S]
    pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    pe = pe.flatten(2).transpose(-1, -2).contiguous()       # [1, 12*16, D]
    model.pos_embed = nn.Parameter(pe)                      # 去掉 cls 后的 pos_embed
    model.pos_hw = (12, 16)                                 # 记录原始二维网格
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = DeiT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_base_patch16_224-b5f2ef4d.pth', map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=False)

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    side = int(np.sqrt(pe.shape[2]))
    pe = pe.view(pe.shape[0], pe.shape[1], side, side)
    pe = F.interpolate(pe, size=(12, 16), mode='bilinear', align_corners=True)
    pe = pe.flatten(2).transpose(-1, -2).contiguous()
    model.pos_embed = nn.Parameter(pe)
    model.pos_hw = (12, 16)
    model.head = nn.Identity()
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = DeiT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load('pretrained/deit_base_patch16_384-8de9b5d1.pth', map_location='cpu')
        model.load_state_dict(ckpt["model"], strict=False)

    pe = model.pos_embed[:, 1:, :].detach()
    pe = pe.transpose(-1, -2)
    side = int(np.sqrt(pe.shape[2]))
    pe = pe.view(pe.shape[0], pe.shape[1], side, side)
    pe = F.interpolate(pe, size=(24, 32), mode='bilinear', align_corners=True)
    pe = pe.flatten(2).transpose(-1, -2).contiguous()
    model.pos_embed = nn.Parameter(pe)
    model.pos_hw = (24, 32)
    model.head = nn.Identity()
    return model
