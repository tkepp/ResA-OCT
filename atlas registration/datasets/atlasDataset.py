#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset of a median patient volume.

"""

from torch.nn.functional import one_hot
from monai.data import NibabelReader
import torch
from torch.utils.data import Dataset
from utilities.Patches import  get_coords, remap_labels, \
    rand_low_res_patch


class MedianAtlasDataset(Dataset):

    def __init__(self, path,  fold:int=0, mode:str=None, patch_size=(77,124,128)):
        """
        Dataset of a median patient volume.

        :param fold: Fold number of representation.
        :param patch_size: Patch size.
        :param mode: Mode of transformation.
        """

        self.reader = NibabelReader()


        self.data = None

        self.patch_mode = mode

        self.patch_size = patch_size


        #get meta
        with torch.no_grad():

            # Load fixed image
            img = torch.load(f"{path}/median_pat_oct_fold_{fold}.pt")

            label = torch.load(f"{path}/median_pat_label_fold_{fold}.pt")

            coordinates = get_coords(img.shape)

            self.data = {'img' : img.squeeze().detach().clone(), 'label' : label.detach().clone(), 'coordinates': coordinates.squeeze()}



    def __len__(self):
        return 1

    def __getitem__(self, idx):

        # get item
        img = self.data['img']
        label = self.data['label'].argmax(dim=-1).squeeze()
        coordinates = self.data['coordinates']


        # choose patches
        if self.patch_mode == 'downsample':

            coordinates = coordinates.permute(3,0,1,2)


            vol = torch.cat([img.unsqueeze(0), label.unsqueeze(0),coordinates], dim=0)

            # compute patch
            with torch.no_grad():
                patch = rand_low_res_patch(vol, patch_size= self.patch_size)

                image_patch = patch[0,...].squeeze()
                coords_patch = patch[2:,...].permute(1, 2, 3, 0).squeeze()

                label_patch = patch[1,...].squeeze().type(torch.LongTensor)
                label_patch = one_hot(label_patch, num_classes=10).float().squeeze()

        elif self.patch_mode is None:
            image_patch = img.squeeze()
            label_patch = label.squeeze().type(torch.LongTensor)
            label_patch = one_hot(label_patch, num_classes=10).float().squeeze()
            coords_patch = coordinates.squeeze()
        else:
            raise NotImplementedError("Patch mode not implemented!")


        image_patch.requires_grad = True
        label_patch.requires_grad = True
        coords_patch.requires_grad = True


        return {'img': image_patch, 'label': label_patch, 'coordinates': coords_patch}

