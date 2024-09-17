# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg, VisionTransformer

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

class MyAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mask_ratio=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, N, C = x.shape
        #print('B N C:', B, N, C)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #print("atth:", attn.shape, 'mean:', attn.mean(), 'min:', attn.min(), 'max:', attn.max())
        #print('attn shape:', attn.shape)

        # token drop
        # Find the 10th percentile value for each attention map
        
        new_attn = attn.view(B, self.num_heads, N*N)
        #print("new atth:", new_attn.shape, new_attn.dtype)#float 32
        threshold = torch.quantile(new_attn, self.mask_ratio, dim=2, keepdim=True) 
        #threshold = torch.mean(new_attn, dim=2, keepdim=True) 
    
        # Create a mask identifying values below the threshold
        mask = new_attn < threshold

        # Apply the mask to zero out values below the threshold
        atten_zeroed = new_attn.masked_fill(mask, 0)
        #print('zero shape', atten_zeroed.shape)
        new_attn = atten_zeroed.reshape(B, self.num_heads, N, N)
        #print("new atth:", new_attn.shape, 'mean:', new_attn.mean(), 'min:', new_attn.min(), 'max:', new_attn.max())
        attn = new_attn

        ############################################

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# Create the Vision Transformer model with the custom attention module
class MyVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default Attention module with your custom one
        m_ratio = 0.65
        print('drop ratio:', m_ratio)
        for blk in self.blocks:
            blk.attn = MyAttention(dim=768, num_heads=blk.attn.num_heads, qkv_bias=True, mask_ratio=m_ratio)

@register_model
def mydeit_tiny_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def mydeit_base_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model