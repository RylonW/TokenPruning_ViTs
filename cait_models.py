# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath

from torch import linalg as LA
__all__ = [
    'cait_M48', 'cait_M36',
    'cait_S36', 'cait_S24','cait_S24_224',
    'cait_XS24','cait_XXS24','cait_XXS24_224',
    'cait_XXS36','cait_XXS36_224'
]

class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x ):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        return x_cls     
        
class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    
    def forward(self, x, x_cls):
        
        u = torch.cat((x_cls,x),dim=1)
        
        
        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
        
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        
        return x_cls 
        
        
class Attention_talking_head(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.proj = nn.Linear(dim, dim)
        
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)
        
        self.proj_drop = nn.Dropout(proj_drop)


    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 
    
        attn = (q @ k.transpose(-2, -1)) 
        
        attn = self.proj_l(attn.permute(0,2,3,1)).permute(0,3,1,2)
                
        attn = attn.softmax(dim=-1)
  
        attn = self.proj_w(attn.permute(0,2,3,1)).permute(0,3,1,2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class LayerScale_Block(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
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
    
class Dropx_LayerScale_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,Attention_block = Attention_talking_head,
                 Mlp_block=Mlp,init_values=1e-4, i=None, drop_ratio=0.1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.drop_ratio=drop_ratio
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
    def forward(self, x):        
        x = self.random_drop_x(x, self.drop_ratio)
        #x = self.drop_low_l2_norm(x, self.drop_ratio, True)
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x 
    
    
    
class cait_models(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to adapt to our cait models
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers = LayerScale_Block,
                 block_layers_token = LayerScale_Block_CA,
                 Patch_layer=PatchEmbed,act_layer=nn.GELU,
                 Attention_block = Attention_talking_head,Mlp_block=Mlp,
                init_scale=1e-4,
                Attention_block_token_only=Class_Attention,
                Mlp_block_token_only= Mlp, 
                depth_token_only=2,
                mlp_ratio_clstk = 4.0):
        super().__init__()
        

            
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)] 
        blk_list = [
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale)
            for i in range(depth)]
        print('drop', 0.1 , 'at', 6)
        blk_list[6] = Dropx_LayerScale_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[6], norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block,Mlp_block=Mlp_block,init_values=init_scale,drop_ratio=0.1)
        self.blocks = nn.ModuleList(blk_list)
        

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer,Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only,init_values=init_scale)
            for i in range(depth_token_only)])
            
        self.norm = norm_layer(embed_dim)


        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i , blk in enumerate(self.blocks):
            x = blk(x)
            
        for i , blk in enumerate(self.blocks_token_only):
            cls_tokens = blk(x,cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)
            
                
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        
        x = self.head(x)

        return x 
        
@register_model
def cait_XXS24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_XXS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 
@register_model
def cait_XXS36_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_XXS36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=192, depth=36, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_XS24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=288, depth=24, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XS24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 




@register_model
def cait_S24_224(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 224,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_S24(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=24, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S24_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model 

@register_model
def cait_S36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384,patch_size=16, embed_dim=384, depth=36, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/S36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 





@register_model
def cait_M36(pretrained=False, **kwargs):
    model = cait_models(
        img_size= 384, patch_size=16, embed_dim=768, depth=36, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M36_384.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)

    return model 


@register_model
def cait_M48(pretrained=False, **kwargs):
    model = cait_models(
        #img_size= 448 , 
        patch_size=16, embed_dim=768, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-6,
        depth_token_only=2,**kwargs)
    
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load('output/Cait448/Cait_448.pth', map_location='cpu')
        '''
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/M48_448.pth",
            map_location="cpu", check_hash=True
        )
        '''
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.'+k]
            
        model.load_state_dict(checkpoint_no_module)
        
    return model         
