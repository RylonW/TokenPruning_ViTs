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
from torchvision.utils import make_grid as make_grid
import matplotlib.pyplot as plt
import numpy as np

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        print('attn drop:', attn_drop )
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

class myAttention(nn.Module):
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


        def random_drop_elements(input_tensor, v_tensor, drop_percentage = 0.):
            """
            Randomly drops elements within each head of the input tensor to achieve a target shape.

            Args:
                input_tensor: The input tensor of shape [batch_size, num_heads, seq_len, hidden_dim].

            Returns:
                The output tensor with randomly dropped elements, maintaining the original batch_size, num_heads, and hidden_dim.
            """

            batch_size, num_heads, seq_len, hidden_dim = input_tensor.shape
            # target_seq_len : The desired sequence length after dropping elements
            print(drop_percentage)
            target_seq_len = seq_len - int((seq_len) * drop_percentage)
            print('target_seq_len:', target_seq_len)

            # Calculate the mean value across the hidden dimension for each element in each head
            mean_values = input_tensor.mean(dim=-1) #[Batch, Heads, Token_num])
            #print('mean shape:', mean_values.shape)

            # Create a mask to indicate which elements to keep for each head
            low0 = False
            low = True
            rand = False
            high = False
            norm_num = 2
            if(low0):
                # Keep the first token
                first_token = input_tensor[:, :, 0:1, :]

                # Apply dropping to the remaining tokens based on their values
                remaining_tokens = input_tensor[:, :, 1:, :]
                '''
                img = remaining_tokens[:,0,:,0].reshape((batch_size, 28,28)).unsqueeze(1)
                print('img', img.shape, img[0,0,0,:])# torch black->low value white->high value
                img = make_grid(img)
                save_image(img, './output/before_filter.jpg')
                #plt.imshow(np.array((img*255).cpu(), dtype=np.uint8))
                #plt.savefig('./output/before_fil.jpg')

                #return 0
                '''
                #values = torch.norm(remaining_tokens, dim=-1)  # Calculate norms (magnitudes)\
                values = torch.norm(remaining_tokens, p=1, dim=3, keepdim=False)/hidden_dim# normalize to [-1,1]
                print('values:', values.shape)
                save_image(make_grid(values[:,0,:].reshape((batch_size, 28,28)).unsqueeze(1)), './output/value.jpg')

                _, indices = torch.topk(values, target_seq_len - 1, dim=2)#[96, 12, 706]
                save_image(make_grid(indices[:,0,:].reshape((batch_size, 28,28)).unsqueeze(1)), './output/topk.jpg')
                #print(indices.shape)
                indices, _ = torch.sort(indices.cuda())# keep original order

                selected_tokens = torch.gather(remaining_tokens, 2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim))

                # Concatenate the first token with the selected tokens
                output_tensor = torch.cat([first_token, selected_tokens], dim=2)
            elif(low):
                # Keep the first token
                first_token = input_tensor[:, :, 0:1, :]

                # Apply dropping to the remaining tokens based on their values
                remaining_tokens = input_tensor[:, :, 1:, :]

                reshaped_tensor = remaining_tokens.reshape(batch_size * num_heads, seq_len-1, hidden_dim)
                #print('high reshape:', reshaped_tensor.shape)
                values = torch.norm(reshaped_tensor, p=norm_num, dim=2, keepdim=False)/hidden_dim
                #print('high values:', values.shape)

                # Sort along the sequence dimension, keeping track of the original indices
                sorted_tensor, indices = torch.sort(values, dim=1, descending=True)# value from big to small
                #print('low sorted tensor:', sorted_tensor)
                print('low indice:', indices.shape)
                
                # clip -> sort( previous sort -> clip)
                # Select the top 'output_seq_len' indices for each head and item
                indices = indices[:, :target_seq_len-1]
                #print('before sort', indices)
                indices, _ = torch.sort(indices.cuda()) # 0->infinite   
                #print('after sort:', indices)            
                
                print('low indices:', indices.shape)

                # Gather the original values based on the selected indices
                output_tensor = torch.gather(reshaped_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim))

                # Reshape back to the original format
                selected_tokens = output_tensor.reshape(batch_size, num_heads, target_seq_len-1, hidden_dim)
                print('low select:', selected_tokens.shape)

                # Concatenate the first token with the selected tokens
                output_tensor = torch.cat([first_token, selected_tokens], dim=2)                
            elif(high):
                # Keep the first token
                first_token = input_tensor[:, :, 0:1, :]

                # Apply dropping to the remaining tokens based on their values
                remaining_tokens = input_tensor[:, :, 1:, :]

                reshaped_tensor = remaining_tokens.reshape(batch_size * num_heads, seq_len-1, hidden_dim)
                #print('high reshape:', reshaped_tensor.shape)
                values = torch.norm(reshaped_tensor, p=norm_num, dim=2, keepdim=False)/hidden_dim
                #print('high values:', values.shape)

                # Sort along the sequence dimension, keeping track of the original indices
                sorted_tensor, indices = torch.sort(values, dim=1, descending=False)# value from small to big
                print('high indice:', indices.shape)
                   
                # clip -> sort( previous sort -> clip)
                # Select the top 'output_seq_len' indices for each head and item
 
                indices = indices[:, :target_seq_len-1]
                #print('before sort', indices)
                indices, _ = torch.sort(indices) # 0->infinite 
                #indices = indices.cuda()
                #print('after sort:', indices)            
                
                print('high indices:', indices.shape)

                # Gather the original values based on the selected indices
                output_tensor = torch.gather(reshaped_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
                #indices.cpu() # move back to cpu to avoid memory leak

                # Reshape back to the original format
                selected_tokens = output_tensor.reshape(batch_size, num_heads, target_seq_len-1, hidden_dim)
                print('high select:', selected_tokens.shape)

                # Concatenate the first token with the selected tokens
                print(first_token.device, selected_tokens.device)
                output_tensor = torch.cat([first_token, selected_tokens], dim=2)
                '''
                # process v
                # Keep the first token
                v_first_token = v_tensor[:, :, 0:1, :]
                # Apply dropping to the remaining tokens based on their values
                v_remaining_tokens = v_tensor[:, :, 1:, :]
                v_reshaped_tensor = v_remaining_tokens.reshape(batch_size * num_heads, seq_len-1, hidden_dim)
                # Gather the original values based on the selected indices
                v_output_tensor = torch.gather(v_reshaped_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
                # Reshape back to the original format
                v_selected_tokens = v_output_tensor.reshape(batch_size, num_heads, target_seq_len-1, hidden_dim)
                print('v high select:', selected_tokens.shape)

                # Concatenate the first token with the selected tokens
                print(first_token.device, selected_tokens.device)
                v_output_tensor = torch.cat([v_first_token, v_selected_tokens], dim=2)
                '''
            elif(rand):# rand in sample
                # Keep the first token
                first_token = input_tensor[:, :, 0:1, :]# Remain cls token

                # Apply random dropping to the remaining tokens
                remaining_tokens = input_tensor[:, :, 1:, :]
                mask = torch.rand(batch_size, num_heads, seq_len - 1)
                _, indices = torch.topk(mask, target_seq_len - 1, dim=2)# value big -> small
                indices,_ = torch.sort(indices.cuda()) # sort ascend
                indices = indices.cuda()# no sort

                selected_tokens = torch.gather(remaining_tokens, 2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim))

                # Concatenate the first token with the selected tokens
                output_tensor = torch.cat([first_token, selected_tokens], dim=2)

                # process v
                # Keep the first token
                v_first_token = v_tensor[:, :, 0:1, :]
                # Apply dropping to the remaining tokens based on their values
                v_remaining_tokens = v_tensor[:, :, 1:, :]
                v_selected_tokens = torch.gather(v_remaining_tokens, 2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim))
                v_output_tensor = torch.cat([v_first_token, v_selected_tokens], dim=2)
            else:# Rand in bacth
                keep_indices = [
                torch.cat((torch.tensor([0]), 
                torch.sort(torch.randperm(seq_len - 1)[:target_seq_len-1] + 1)[0]))
                            for _ in range(num_heads)]
                #print(keep_indices[0].shape, keep_indices[0])
                #print(keep_indices[0].shape)

                # Apply the mask to each head and gather the selected elements
                output_tensor = torch.stack([
                    input_tensor[:, head_idx, keep_indices[head_idx], :] 
                    for head_idx in range(num_heads)
                ], dim=1)

            if(high or low):
                # process v
                # Keep the first token
                v_first_token = v_tensor[:, :, 0:1, :]
                # Apply dropping to the remaining tokens based on their values
                v_remaining_tokens = v_tensor[:, :, 1:, :]
                v_reshaped_tensor = v_remaining_tokens.reshape(batch_size * num_heads, seq_len-1, hidden_dim)
                # Gather the original values based on the selected indices
                v_output_tensor = torch.gather(v_reshaped_tensor, 1, indices.unsqueeze(-1).expand(-1, -1, hidden_dim))
                # Reshape back to the original format
                v_selected_tokens = v_output_tensor.reshape(batch_size, num_heads, target_seq_len-1, hidden_dim)
                print('v high select:', selected_tokens.shape)

                # Concatenate the first token with the selected tokens
                print(first_token.device, selected_tokens.device)
                v_output_tensor = torch.cat([v_first_token, v_selected_tokens], dim=2)

            return output_tensor, v_output_tensor
        new_k, new_v = random_drop_elements(k, v)
        print('k:', new_k.shape)
        #return 0
        #new_v = random_drop_elements(v)


        attn = (q @ new_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ new_v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# My custom block  
class myBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_ratio=0.01):
        super().__init__()
        self.norm1 = norm_layer(dim)
        change_attn = True
        if(change_attn):
            self.attn = myAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        else:
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
        num_heads = 12
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

    def random_drop_x(self, x, drop_percentage):
        B, N, C = x.shape

        # Create the tensor
        tensor = torch.ones((B, N-1), dtype=torch.bool).cuda()  # Initialize with True

        # Vectorized random index selection
        threshold = int((N-1) * drop_percentage)
        random_indices = torch.argsort(torch.rand(tensor.shape).argsort(dim=1), dim=1)[:, :threshold].cuda()

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
        
        return output, final_mask 

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
        sort = torch.argsort(torch.argsort(l2_norms, dim=1), dim=1) # descend
        #threshold = int((N-1) * drop_percentage)
        threshold = int((N-1) * (1-drop_percentage))
        #print(sort)

        # Create a mask to keep elements above the threshold
        #mask = sort > threshold
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
        return output, final_mask


    def forward(self, x):
        #print('block input shape:', x.shape, 'attn shape:', self.attn(self.norm1(x)).shape)
        # drop x
        #self.save_tmp(x)
        #return 0
        #x, _ = self.drop_low_l2_norm(x, self.drop_ratio, True)
        #x, _ = self.drop_high_l2_norm(x, self.drop_ratio)
        #x, _, _, _ = self.drop_low_attn(x, self.drop_ratio, True, self.attn.qkv)
        #x, _ = self.random_drop_x(x, self.drop_ratio)
        #print('drop x shape:', x.shape)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def save_tmp(self, x):
        # Low - L2 Norm
        torch.save(x, '/home/ruilu/TokenPruning_ViTs/output/features/x.pt')
        drop_low_l2_norm, drop_low_l2_norm_mask = self.drop_low_l2_norm(x, self.drop_ratio, True)
        torch.save(drop_low_l2_norm, '/home/ruilu/TokenPruning_ViTs/output/features/drop5_lownorm.pt')
        torch.save(drop_low_l2_norm_mask, '/home/ruilu/TokenPruning_ViTs/output/features/drop5_lownorm_mask.pt')
        
        # Low - Q/K
        drop_low_q, drop_low_q_mask, drop_low_q_l2, cls_attn = self.drop_low_attn(x, self.drop_ratio, True, self.attn.qkv)
        torch.save(drop_low_q, '/home/ruilu/TokenPruning_ViTs/output/features/drop_low_q.pt')
        torch.save(drop_low_q_mask, '/home/ruilu/TokenPruning_ViTs/output/features/drop_low_q_mask.pt')
        torch.save(drop_low_q_l2, '/home/ruilu/TokenPruning_ViTs/output/features/drop_low_q_l2.pt')
        torch.save(cls_attn, '/home/ruilu/TokenPruning_ViTs/output/features/drop_low_q_clsattn.pt')
        
        # Low - Random
        drop_low_rand, drop_low_rand_mask = self.random_drop_x(x, self.drop_ratio)
        torch.save(drop_low_rand, '/home/ruilu/TokenPruning_ViTs/output/features/drop_low_rand.pt')
        torch.save(drop_low_rand_mask, '/home/ruilu/TokenPruning_ViTs/output/features/drop_low_rand_mask.pt')
        
        # High - L2 Norm
        drop_high_l2_norm, drop_high_l2_norm_mask = self.drop_high_l2_norm(x, self.drop_ratio)
        torch.save(drop_high_l2_norm, '/home/ruilu/TokenPruning_ViTs/output/features/drop5_highnorm.pt')
        torch.save(drop_high_l2_norm_mask, '/home/ruilu/TokenPruning_ViTs/output/features/drop5_high_mask.pt')
        return 0

# Create the Vision Transformer model with the custom attention module
class MyVisionTransformer(VisionTransformer):
    def __init__(self, blk_num, drop_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace the default Attention module with your custom one
        print('trash', args, kwargs)
        for idx, blk in enumerate(self.blocks):
            if(idx==blk_num):
                print('my block', idx)
                self.blocks[idx] = myBlock(
                #dim=768, num_heads=12, mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_ratio=drop_ratio)
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=4, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_ratio=drop_ratio)


@register_model
def pxdeit_small_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_num=7, drop_ratio=0.3, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def pxdeit_base_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_num=7, drop_ratio=0.3, **kwargs)
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
        #patch_size=16, embed_dim=768, depth=48, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_ratio=0.,  blk_num=4,  **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def pxdeit_base_patch16_448(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        #img_size=448, 
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_num=4, drop_ratio=0.1,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def pxdeit_base_patch16_512(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        #img_size=512, 
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), blk_num=7, drop_ratio=0.1, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model

