#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
import torch


def rand_low_res_patch(volume, patch_size = (33,62,64)):
    """"
    Method to sample one single random 3D patch from a voluem.

    :param volume: Tensor of volumes (N, H, W, C).
    :param patch_size: Size of patches to sample.
    :param num_patches: Number of random patches to sample.
    """
    _, h,w,d  = volume.shape

    step_h = h // patch_size[0]
    step_w = w // patch_size[1]
    step_d = d // patch_size[2]


    h_ind = []
    for shift in range(step_h):
        h_ind.append(torch.arange(shift, h, step_h)[:patch_size[0]])

    w_ind = []
    for shift in range(step_w):
        w_ind.append(torch.arange(shift, w, step_w)[:patch_size[1]])

    d_ind = []
    for shift in range(step_d):
        d_ind.append(torch.arange(shift, d, step_d)[:patch_size[2]])

    prods = list(product(h_ind, w_ind, d_ind))
    id = np.random.randint(0, len(prods))

    hi, wi, di = prods[id]
    # build a 3D mesh of coords and index in one go
    H_idx, W_idx, D_idx = torch.meshgrid(hi, wi, di, indexing='ij')
    # each of H_idx, W_idx, D_idx has shape (ph, pw, pd)
    patch = volume[..., H_idx, W_idx, D_idx]  # (ph, pw, pd)


    return patch


def get_coords(tensor_size=(10, 10, 10)):
    """
    Method to get coordinates of input tensor size.

    :param tensor_size: (H,W,D)
    :return: (H,W,D,3) coordinates of input tensor size. Coordinate vector is structured (x,y,z)
    """
    x,y, z = torch.meshgrid(torch.linspace(-1, 1, int(tensor_size[0])),
                             torch.linspace(-1, 1, int(tensor_size[1])),
                             torch.linspace(-1, 1, int(tensor_size[2])), indexing='ij')

    coords = torch.stack([x, y, z], dim=-1)
    return coords

def remap_labels(seg):
    """Method to summarise small classes"""
    out_seg = np.zeros_like(seg)
    out_seg[seg == 1] = 1
    out_seg[seg == 2] = 2
    out_seg[seg == 3] = 3
    out_seg[seg == 4] = 4
    out_seg[seg == 5] = 5
    out_seg[seg == 6] = 6
    out_seg[seg == 7] = 7
    out_seg[seg == 8] = 7
    out_seg[seg == 9] = 8
    out_seg[seg == 10] = 8
    out_seg[seg == 11] = 9

    return out_seg
