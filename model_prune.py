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

from torchvision.utils import save_image as save_image
import matplotlib.pyplot as plt

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

    def drop_low_l2_norm(self, x, drop_percentage, scale):
        """Calculates L2 norm along dim 1, drops lowest percentile, and reshapes.

        Args:
            x (torch.Tensor): Input tensor with shape [1, 197, 192].
            drop_percentage (float): Percentage of elements to drop (default: 0.10).

        Returns:
            torch.Tensor: Output tensor with shape [1, num_remaining_elements, 192].
        """
        B, N, C = x.shape
        #print(B,N,C)
        # Calculate L2 norm along dimension 1
        #print('patch 1:', x[0,1,:], x[0,1,:].shape)
        l2_norms = LA.norm(x[:,1:,:], ord=2, dim=2) / C # Shape: [B, N-1] cls token should not be deleted
        #print('l2_norm:', l2_norms.shape)

        # Find the threshold for the bottom 10%
        #threshold = torch.quantile(l2_norms, drop_percentage)
        sort = torch.argsort(l2_norms, dim=1)
        threshold = int((N-1) * (1-drop_percentage))
        #print(sort)

        # Create a mask to keep elements above the threshold
        mask = sort < threshold  # Shape: [1, 192]
        #print(mask.shape)
        #print('mask:', mask, mask.shape)

        # Expand the mask to match the shape of x
        #expanded_mask = mask.unsqueeze(2).expand_as(x)  # Shape: [1, 197, 192]
        #new_N = expanded_mask.shape[0]/B/C
        #print(new_N)
        expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]

        #print('expanded_mask', expanded_mask.shape, )
        expanded_mask = expanded_mask.reshape(B, N-1, C)
        cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
        final_mask = torch.cat((cls_mask, expanded_mask), dim=1)
        #print('final mask:', final_mask.shape)
        #print('exp mask 0:', expanded_mask[:,0,:], expanded_mask[:,0,:].shape)        
        
        # Apply the expanded mask
        remaining_elements = x[final_mask]  # Shape: [num_remaining, 192]
        
        #print('remaining:', remaining_elements.shape, remaining_elements.shape[0]/B/C)
        new_N = int(remaining_elements.shape[0]/B/C)

        # Adjust the mask to drop extra elements if necessary
        #if actual_remaining_elements > num_remaining_elements:
        #    remaining_elements = remaining_elements[:num_remaining_elements * 192]
        output = remaining_elements.view(B, new_N, C)
        merge = True
        if(merge):
            # Mereg
            merge_mask = sort > threshold
            expanded_merge_mask = torch.repeat_interleave(merge_mask, C, dim=1)
            expanded_merge_mask = expanded_merge_mask.reshape(B, N-1, C)
            merge_cls_mask = torch.zeros((B, 1, C), dtype=torch.bool).cuda()
            merge_cls_mask = torch.cat((merge_cls_mask, expanded_merge_mask), dim=1)
            merge_elements = x[merge_cls_mask] #[B, ~, C]
            #print(merge_elements.shape)
            merge_N = int(merge_elements.shape[0]/B/C)
            merge_elements = merge_elements.view(B, merge_N, C)
            merge_avg = False
            if(merge_avg):
                y = torch.mean(merge_elements, 1, True)
            else:
                y = torch.sum(merge_elements, 1, True)
            output = torch.cat((output, y), dim=1)
            #print(output.shape)
        if(scale):          
            output = output / (1 - drop_percentage)
        #print(x.sum(), output.sum())
        return output

    def drop_low_attn(self, x, drop_percentage, scale, qkv_layer):
        B, N, C = x.shape
        num_heads = 12
        #qkv = qkv_layer(self.norm1(x)).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        qkv = qkv_layer(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #[B, num_head, N, 64]
        # selct k / q/ v
        tmp = k
        tmp = torch.mean(tmp, 1) # average 12 heads
        _,_,k_C = tmp.shape

        # qkv
        l2_norms = LA.norm(tmp[:,1:,:], ord=2, dim=2) / k_C # Shape: [B, N-1] cls token should not be deleted
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
        
        # Adjust the mask to drop extra elements if necessary
        new_N = int(remaining_elements.shape[0]/B/C)
        output = remaining_elements.view(B, new_N, C)
        if(scale):          
            output = output / (1 - drop_percentage)
        return output

    def random_drop_x(self, x, drop_percentage):
        B, N, C = x.shape
        #print(B,N,C)
        # Create the tensor
        tensor = torch.ones((B, N-1), dtype=torch.bool).cuda()  # Initialize with True

        # Vectorized random index selection
        threshold = int((N-1) * drop_percentage)
        random_indices = torch.rand(tensor.shape).argsort(dim=1)[:, :threshold].cuda()

        # Vectorized setting to False
        tensor.scatter_(1, random_indices, False)
        #print(tensor[0,:])
        mask = tensor

        # Expand the mask to match the shape of x
        #expanded_mask = mask.unsqueeze(2).expand_as(x)  # Shape: [1, 197, 192]
        #new_N = expanded_mask.shape[0]/B/C
        #print(new_N)
        expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]

        #print('expanded_mask', expanded_mask.shape, )
        expanded_mask = expanded_mask.reshape(B, N-1, C)
        cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
        final_mask = torch.cat((cls_mask, expanded_mask), dim=1)
        #print('final mask:', final_mask.shape)
        #print('exp mask 0:', expanded_mask[:,0,:], expanded_mask[:,0,:].shape)

        # Apply the expanded mask
        remaining_elements = x[final_mask]  # Shape: [num_remaining, 192]
        #print('remaining:', remaining_elements.shape, remaining_elements.shape[0]/B/C)
        new_N = int(remaining_elements.shape[0]/B/C)


        # Adjust the mask to drop extra elements if necessary
        #if actual_remaining_elements > num_remaining_elements:
        #    remaining_elements = remaining_elements[:num_remaining_elements * 192]
        output = remaining_elements.view(B, new_N, C)
        l2_norms = LA.norm(output, ord=2, dim=2) / C # Shape: [B, 197]
        #print('l2_norm:', l2_norms, l2_norms.shape)
        #print('patch 1:', output[0,0,:], output[0,0,:].shape)

        # Reshape to the desired output shape
        #return remaining_elements.view(1, num_remaining_elements, 192)
        
        return output    

    def drop_high_l2_norm(self, x, drop_percentage):
        """Calculates L2 norm along dim 1, drops highest percentile, and reshapes.

        Args:
            x (torch.Tensor): Input tensor with shape [1, 197, 192].
            drop_percentage (float): Percentage of elements to drop (default: 0.10).

        Returns:
            torch.Tensor: Output tensor with shape [1, num_remaining_elements, 192].
        """
        B, N, C = x.shape
        #print(B,N,C)
        # Calculate L2 norm along dimension 1
        #print('patch 1:', x[0,1,:], x[0,1,:].shape)
        l2_norms = LA.norm(x[:,1:,:], ord=2, dim=2) / C # Shape: [B, N-1] cls token should not be deleted
        #print('l2_norm:', l2_norms.shape)

        # Find the threshold for the bottom 10%
        #threshold = torch.quantile(l2_norms, drop_percentage)
        sort = torch.argsort(l2_norms, dim=1) # descend
        threshold = int((N-1) * drop_percentage)
        #print(sort)

        # Create a mask to keep elements above the threshold
        mask = sort > threshold  # Shape: [1, 192]
        #print(mask.shape)
        #print('mask:', mask, mask.shape)

        # Expand the mask to match the shape of x
        #expanded_mask = mask.unsqueeze(2).expand_as(x)  # Shape: [1, 197, 192]
        #new_N = expanded_mask.shape[0]/B/C
        #print(new_N)
        expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]

        #print('expanded_mask', expanded_mask.shape, )
        expanded_mask = expanded_mask.reshape(B, N-1, C)
        cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
        final_mask = torch.cat((cls_mask, expanded_mask), dim=1)
        #print('final mask:', final_mask.shape)
        #print('exp mask 0:', expanded_mask[:,0,:], expanded_mask[:,0,:].shape)

        # Apply the expanded mask
        remaining_elements = x[final_mask]  # Shape: [num_remaining, 192]
        #print('remaining:', remaining_elements.shape, remaining_elements.shape[0]/B/C)
        new_N = int(remaining_elements.shape[0]/B/C)


        # Adjust the mask to drop extra elements if necessary
        #if actual_remaining_elements > num_remaining_elements:
        #    remaining_elements = remaining_elements[:num_remaining_elements * 192]
        output = remaining_elements.view(B, new_N, C)
        return output

    def draw(self,x, drop_x):
        B,N,C = x.shape
        #print(B,N,C, x[:,1:,:].mean(2, True).shape, x[:,1:,:].mean(2, True).view(B, 1,14,14).shape)
        vis = x[:,1:,:].mean(2, True).view(B, 1,14,14)
        vis -= vis.min()
        vis /= vis.max()
        vis = vis*255
        #print(vis[0,:,:,:])
        save_image(vis, 'output/vis/0.jpg')
        plt.imsave('output/vis/0_1.jpg', vis[0,:,:,:].squeeze().cpu())

        print(drop_x.shape)
        #print(vis[0,:,:,:])
        save_image(vis, 'output/vis/drop.jpg')
        plt.imsave('output/vis/drop_1.jpg', vis[0,:,:,:].squeeze().cpu())
        return 0
    def forward(self, x):
        #print('block input shape:', x.shape, 'attn shape:', self.attn(self.norm1(x)).shape)
        # drop x
        x = self.drop_low_l2_norm(x, self.drop_ratio, True)
        #x = self.drop_high_l2_norm(x, self.drop_ratio)
        #x = self.drop_low_attn(x, self.drop_ratio, True, self.attn.qkv)
        #x = self.random_drop_x(x, self.drop_ratio)
        #print('drop x shape:', x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# Create the Vision Transformer model with the custom attention module
class MyVisionTransformer(VisionTransformer):
    def __init__(self, blk_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default Attention module with your custom one
        print('trash')
        for idx, blk in enumerate(self.blocks):
            if(idx==blk_num):
                print('my block', idx)
                self.blocks[idx] = myBlock(
                dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_ratio=0.1)
                #blk.attn = MyAttention(dim=192, num_heads=blk.attn.num_heads, qkv_bias=True)

@register_model
def pxdeit_base_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_num=1,  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pxdeit_base_patch16_384(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_num=7,  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

