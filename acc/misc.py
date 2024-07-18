# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import datetime
import math
import os
import time
from collections import defaultdict, deque, OrderedDict

import logging as logging
import psutil
import torch
import torch.distributed as dist
#import torch.fb.rendezvous.zeus
from iopath.common.file_io import g_pathmgr as pathmgr
#from logging import master_print as print
from torch import inf


#logger = logging.get_logger(__name__)




class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, fp32=False):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not fp32)

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    checkpoint_path = "{}/checkpoint-{:05d}.pth".format(args.output_dir, epoch)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scaler": loss_scaler.state_dict(),
        "args": args,
    }

    save_on_master(to_save, checkpoint_path)
    return checkpoint_path


def get_last_checkpoint(args):
    """
    Get the last checkpoint from the checkpointing folder.
    Args:
        path_to_job (string): the path to the folder of the current job.
    """
    d = args.output_dir
    names = pathmgr.ls(d) if pathmgr.exists(d) else []
    names = [f for f in names if "checkpoint" in f]
    if len(names) == 0:
        print("No checkpoints found in '{}'.".format(d))
        return None
    else:
        # Sort the checkpoints by epoch.
        name = sorted(names)[-1]
        return os.path.join(d, name)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if not args.resume:
        args.resume = get_last_checkpoint(args)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            with pathmgr.open(args.resume, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        print("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not (hasattr(args, "eval") and args.eval)
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024**3


def cpu_mem_usage():
    """
    Compute the system memory (RAM) usage for the current device (GB).
    Returns:
        usage (float): used memory (GB).
        total (float): total memory (GB).
    """
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024**3
    total = vram.total / 1024**3

    return usage, total


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor


def add_weight_decay(model, weight_decay=1e-5, skip_list=(), bias_wd=False):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if (
            (not bias_wd)
            and len(param.shape) == 1
            or name.endswith(".bias")
            or name in skip_list
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def inflate(model_2d, model_3d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "patch_embed.proj.weight" in k:
            v3d = model_3d[k]
            v3d = v2d.unsqueeze(2).repeat(1, 1, v3d.shape[2], 1, 1) / v3d.shape[2]
            state_dict_inflated[k] = v3d.clone()
        elif "pos_embed" in k:
            pos_embed_cls, pos_embed_spatial = torch.split(v2d, [1, 196], dim=1)
            state_dict_inflated["pos_embed_cls"] = pos_embed_cls.clone()
            state_dict_inflated["pos_embed"] = pos_embed_spatial.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated


def convert_checkpoint(model_2d):
    state_dict_inflated = OrderedDict()
    for k, v2d in model_2d.items():
        if "head.projection.weight" in k:
            state_dict_inflated["head.weight"] = v2d.clone()
        elif "head.projection.bias" in k:
            state_dict_inflated["head.bias"] = v2d.clone()
        else:
            state_dict_inflated[k] = v2d.clone()
    return state_dict_inflated