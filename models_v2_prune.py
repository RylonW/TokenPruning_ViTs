# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch import linalg as LA
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model



class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
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
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class myAttention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_percentage=0.65):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_percentage = drop_percentage

    def forward(self, x):
        #print('attn drop percentage:', self.drop_percentage)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1)) #[B, heads, ~,~]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #---------------------------------------------------
        # token drop
        # Find the 10th percentile value for each attention map
        
        new_attn = attn.view(B, self.num_heads, N*N)
        #print("new atth:", new_attn.shape, new_attn.dtype)#float 32
        threshold = torch.quantile(new_attn, self.drop_percentage, dim=2, keepdim=True) 
        #threshold = torch.mean(new_attn, dim=2, keepdim=True) 
    
        # Create a mask identifying values below the threshold
        mask = new_attn < threshold

        # Apply the mask to zero out values below the threshold
        atten_zeroed = new_attn.masked_fill(mask, 0)
        #print('zero shape', atten_zeroed.shape)
        new_attn = atten_zeroed.reshape(B, self.num_heads, N, N)
        #print("new atth:", new_attn.shape, 'mean:', new_attn.mean(), 'min:', new_attn.min(), 'max:', new_attn.max())
        attn = new_attn

        #---------------------------------------------------
        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 

class Layer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class myLayer_scale_init_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4, drop_ratio=0.3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
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
        sort = torch.argsort(torch.argsort(l2_norms, dim=1), dim=1)
        #threshold = int((N-1) * (1-drop_percentage))
        threshold = int((N-1) * drop_percentage)
        #print(sort)

        # Create a mask to keep elements above the threshold
        #mask = sort < threshold  # Shape: [1, 192]
        mask = sort > threshold
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
        return output, final_mask
    def drop_low_attn(self, x, drop_percentage, scale, qkv_layer):
        B, N, C = x.shape
        num_heads = 16 # large 12# base
        #qkv = qkv_layer(self.norm1(x)).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        qkv = qkv_layer(x).reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] #[B, num_head, N, 64]
        attn = (q @ k.transpose(-2, -1))
        cls_attn = attn #[B, 1025, 1025]
        use_attn_mask =True
        final_mask = None
        l2_norms = None
        if(use_attn_mask):
            #print(cls_attn.shape)
            mean_cls_attn = torch.mean(cls_attn, 1)
            l2_norms = mean_cls_attn[:,0,1:]
            sort = torch.argsort(torch.argsort(mean_cls_attn[:,0,1:], dim=1), dim=1)#larger value->larger num
            threshold = int((N-1) * drop_percentage)
            mask = sort > threshold  # Shape: [1, 192]
            #print('mask shape:', mask.shape)
            # Expand the mask to match the shape of x
            expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]
            expanded_mask = expanded_mask.reshape(B, N-1, C)
            cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
            final_mask = torch.cat((cls_mask, expanded_mask), dim=1)     
        else:
            # selct k / q/ v
            tmp = k
            tmp = torch.mean(tmp, 1) # average 12 heads
            _,_,k_C = tmp.shape

            # qkv (Drop according to qkv)
            l2_norms = LA.norm(tmp[:,1:,:], ord=2, dim=2) / k_C # Shape: [B, N-1] cls token should not be deleted
            #l2_norms = self.median_filter_reshaped(l2_norms, 3)
            
            # Find the threshold for the bottom 10%
            sort = torch.argsort(torch.argsort(l2_norms, dim=1), dim=1)
            threshold = int((N-1) * drop_percentage)
            # Create a mask to keep elements above the threshold
            mask = sort > threshold  # Shape: [1, 192]
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
        return output, final_mask, l2_norms, cls_attn
    def set_low_to_zero(self, x, drop_percentage, qkv_layer):
        B, N, C = x.shape
        #print(B,N,C)
        x_l2 = False
        if(x_l2):
            # Calculate L2 norm along dimension 1
            l2_norms = LA.norm(x[:,1:,:], ord=2, dim=2) / C # Shape: [B, N-1] cls token should not be deleted

            # Find the threshold for the bottom 10%
            sort = torch.argsort(torch.argsort(l2_norms, dim=1), dim=1)
            # Low |------|(thresh)++++++++++++++| High
            threshold = int((N-1) * drop_percentage)
            #print(sort)

            # Create a mask to keep elements above the threshold
            mask = sort > threshold

            # Expand the mask to match the shape of x
            expanded_mask = torch.repeat_interleave(mask, C, dim=1) #[B,N,C]
            expanded_mask = expanded_mask.reshape(B, N-1, C)
            cls_mask = torch.ones((B, 1, C), dtype=torch.bool).cuda() 
            final_mask = torch.cat((cls_mask, expanded_mask), dim=1)
            final_mask = ~final_mask # reverse, so that False elements are set to 0     
            
            # Apply the expanded mask
            x[final_mask] = 0.
            remaining_elements = x
            new_N = int(remaining_elements.shape[0]/B/C)

            # Adjust the mask to drop extra elements if necessary
            #if actual_remaining_elements > num_remaining_elements:
            output = remaining_elements.view(B, N, C)
        return output
    def forward(self, x):
        #print('block input shape:', x.shape, 'attn shape:', self.attn(self.norm1(x)).shape)
        # drop x
        #self.save_tmp(x)
        #return 0
        #x, _ = self.drop_low_l2_norm(x, self.drop_ratio, True)
        #x, _ = self.drop_high_l2_norm(x, self.drop_ratio)
        #x, _, _, _ = self.drop_low_attn(x, self.drop_ratio, True, self.attn.qkv)
        x, _ = self.random_drop_x(x, self.drop_ratio)
        #x = self.set_low_to_zero(x, self.drop_ratio, self.attn.qkv)
        #print('drop x shape:', x.shape)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class Layer_scale_init_Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_1_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1*self.attn(self.norm1(x))) + self.drop_path(self.gamma_1_1 * self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x))) + self.drop_path(self.gamma_2_1 * self.mlp1(self.norm21(x)))
        return x
        
class Block_paralx2(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention,Mlp_block=Mlp
                 ,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm11 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.attn1 = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm21 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp1 = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) + self.drop_path(self.attn1(self.norm11(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.mlp1(self.norm21(x)))
        return x
        
        
class hMLP_stem(nn.Module):
    """ hMLP_stem: https://arxiv.org/pdf/2203.09795.pdf
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, embed_dim=768,norm_layer=nn.SyncBatchNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = torch.nn.Sequential(*[nn.Conv2d(in_chans, embed_dim//4, kernel_size=4, stride=4),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim//4, kernel_size=2, stride=2),
                                          norm_layer(embed_dim//4),
                                          nn.GELU(),
                                          nn.Conv2d(embed_dim//4, embed_dim, kernel_size=2, stride=2),
                                          norm_layer(embed_dim),
                                         ])
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class vit_models(nn.Module):
    """ Vision Transformer with LayerScale (https://arxiv.org/abs/2103.17239) support
    taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    with slight modifications
    """
    def __init__(self, img_size=224,  patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = Block,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention, Mlp_block=Mlp,
                dpr_constant=True,init_scale=1e-4,
                mlp_ratio_clstk = 4.0,**kwargs):
        super().__init__()
        
        self.dropout_rate = drop_rate

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)])
        

        
            
        self.norm = norm_layer(embed_dim)

        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    
    def get_num_layers(self):
        return len(self.blocks)
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = x + self.pos_embed
        
        x = torch.cat((cls_tokens, x), dim=1)
            
        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):

        x = self.forward_features(x)
        
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        
        return x

# Create the Vision Transformer model with the custom attention module
class MyVisionTransformer(vit_models):
    def __init__(self, blk_num, drop_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default Attention module with your custom one
        print('trash', args, kwargs)
        for idx, blk in enumerate(self.blocks):
            if(idx==blk_num):# drop one layer
                print('my block', idx)
                self.blocks[idx] = myLayer_scale_init_Block(
                #dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_ratio=drop_ratio)
                                                            dim=kwargs['embed_dim']
                                                          , num_heads=kwargs['num_heads']
                                                          , mlp_ratio=4
                                                          , qkv_bias=True
                                                          , norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                          , drop_ratio=drop_ratio)
            elif(blk_num == -1): # drop every layer
                self.blocks[idx] = myLayer_scale_init_Block(dim=kwargs['embed_dim']
                                                          , num_heads=kwargs['num_heads']
                                                          , mlp_ratio=4
                                                          , qkv_bias=True
                                                          , norm_layer=partial(nn.LayerNorm, eps=1e-6)
                                                          , drop_ratio=drop_ratio
                                                          , Attention_block = myAttention)

# DeiT III: Revenge of the ViT (https://arxiv.org/abs/2204.07118)

@register_model
def deit_tiny_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,   **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    
    return model
    
    
@register_model
def deit_small_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_small_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])

    return model

@register_model
def deit_medium_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False, **kwargs):
    model = vit_models(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_medium_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model 

@register_model
def deit_base_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_base_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    
@register_model
def pxdeit_large_patch16_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = MyVisionTransformer(
        img_size = img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block
        , blk_num=-1, drop_ratio=0.01, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_large_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k.pth'
        else:
            name+='1k.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    
@register_model
def deit_huge_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    if pretrained:
        name = 'https://dl.fbaipublicfiles.com/deit/deit_3_huge_'+str(img_size)+'_'
        if pretrained_21k:
            name+='21k_v1.pth'
        else:
            name+='1k_v1.pth'
            
        checkpoint = torch.hub.load_state_dict_from_url(
            url=name,
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
    
@register_model
def deit_huge_patch14_52_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=52, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)

    return model
    
@register_model
def deit_huge_patch14_26x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1280, depth=26, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block_paralx2, **kwargs)

    return model
    
@register_model
def deit_Giant_48x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Block_paral_LS, **kwargs)

    return model

@register_model
def deit_giant_40x2_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Block_paral_LS, **kwargs)
    return model

@register_model
def deit_Giant_48_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1664, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    return model

@register_model
def deit_giant_40_patch14_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers = Layer_scale_init_Block, **kwargs)
    #model.default_cfg = _cfg()

    return model

# Models from Three things everyone should know about Vision Transformers (https://arxiv.org/pdf/2203.09795.pdf)

@register_model
def deit_small_patch16_36_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)

    return model
    
@register_model
def deit_small_patch16_36(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=36, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model
    
@register_model
def deit_small_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model
    
@register_model
def deit_small_patch16_18x2(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=384, depth=18, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block_paralx2, **kwargs)

    return model
    
  
@register_model
def deit_base_patch16_18x2_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block_paralx2, **kwargs)

    return model


@register_model
def deit_base_patch16_18x2(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=18, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Block_paralx2, **kwargs)

    return model
    

@register_model
def deit_base_patch16_36x1_LS(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),block_layers=Layer_scale_init_Block, **kwargs)

    return model

@register_model
def deit_base_patch16_36x1(pretrained=False, img_size=224, pretrained_21k = False,  **kwargs):
    model = vit_models(
        img_size = img_size, patch_size=16, embed_dim=768, depth=36, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model

