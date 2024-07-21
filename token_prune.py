import torch
import torch.nn as nn
from functools import partial
import timm
from timm.models.vision_transformer import VisionTransformer,  _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed, Mlp, DropPath

import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# My custom block  
class myBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_ratio=0.1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_ratio=drop_ratio
        print('drop ratio:', self.drop_ratio)

    def drop_low_l2_norm(self, x, drop_percentage):
        """Calculates L2 norm along dim 1, drops lowest percentile, and reshapes.

        Args:
            x (torch.Tensor): Input tensor with shape [1, 197, 192].
            drop_percentage (float): Percentage of elements to drop (default: 0.10).

        Returns:
            torch.Tensor: Output tensor with shape [1, num_remaining_elements, 192].
        """
        B, N, C = x.shape
        # Calculate L2 norm along dimension 1
        l2_norms = LA.norm(x[:,1:,:], ord=2, dim=2) / C # Shape: [B, N-1] cls token should not be deleted

        # Find the threshold for the bottom 10%
        sort = torch.argsort(l2_norms, dim=1)
        threshold = int((N-1) * (1-drop_percentage))

        # Create a mask to keep elements above the threshold
        mask = sort < threshold  # Shape: [1, 192]


        # Expand the mask to match the shape of x
        expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]
        expanded_mask = expanded_mask.reshape(B, N-1, C)
        cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
        final_mask = torch.cat((cls_mask, expanded_mask), dim=1)

        # Apply the expanded mask
        remaining_elements = x[final_mask]  # Shape: [num_remaining, 192]
        new_N = int(remaining_elements.shape[0]/B/C)


        # Adjust the mask to drop extra elements if necessary
        output = remaining_elements.view(B, new_N, C)

        return output

    def random_drop_x(self, x, drop_percentage):
        B, N, C = x.shape
        # Create the tensor
        tensor = torch.ones((B, N-1), dtype=torch.bool).cuda()  # Initialize with True

        # Vectorized random index selection
        threshold = int((N-1) * drop_percentage)
        random_indices = torch.rand(tensor.shape).argsort(dim=1)[:, :threshold].cuda()

        # Vectorized setting to False
        tensor.scatter_(1, random_indices, False)
        mask = tensor

        # Expand the mask to match the shape of x
        expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]
        expanded_mask = expanded_mask.reshape(B, N-1, C)
        cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
        final_mask = torch.cat((cls_mask, expanded_mask), dim=1)


        # Apply the expanded mask
        remaining_elements = x[final_mask]  # Shape: [num_remaining, 192]
        new_N = int(remaining_elements.shape[0]/B/C)


        # Adjust the mask to drop extra elements if necessary
        output = remaining_elements.view(B, new_N, C)
        return output    

    def forward(self, x):
        #print('block input shape:', x.shape, 'attn shape:', self.attn(self.norm1(x)).shape)
        # drop x
        #x = self.drop_low_l2_norm(x, self.drop_ratio)
        x= self.random_drop_x(x, self.drop_ratio)
        #print('drop x shape:', x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Create the Vision Transformer model with the custom attention module
class MyVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default Attention module with your custom one
        print('trash')
        for idx, blk in enumerate(self.blocks):
            if(idx==6):
                print('my block', idx)
                self.blocks[idx] = myBlock(
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_ratio=0.5)
                #blk.attn = MyAttention(dim=192, num_heads=blk.attn.num_heads, qkv_bias=True)

@register_model
def pxdeit_base_patch16_224(pretrained=False, **kwargs):
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

