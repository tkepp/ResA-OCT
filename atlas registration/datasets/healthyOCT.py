#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main dataset to process healthy OCT data.

"""

from torch.nn.functional import one_hot
from monai.data import NibabelReader
import torch
from torch.utils.data import Dataset
from utilities.Patches import  get_coords, remap_labels, \
    rand_low_res_patch



class HealthyOctDataset(Dataset):

    def __init__(self, path_list:list =None, mode:str=None, patch_size=(231,62,64)):
        """
        Dataset for healthy oct patients.

        :param path_list: list of paths to images.
        """

        if path_list is None:
            raise RuntimeError("Paths are missing!")

        if not isinstance(path_list, list):
            raise RuntimeError("path_list is not a list!")

        self.reader = NibabelReader()

        self.pseudo_id = 0

        self.data_list = []

        self.patch_mode = mode

        self.patch_size = patch_size

        for patient in path_list:

            #get meta
            with torch.no_grad():
                pat_id = patient.split("/")[-3]
                lat = patient.split("/")[-2]

                # Load fixed image
                img_object = self.reader.read(patient)
                img, _ = self.reader.get_data(img_object)

                # convert to torch and shape (H,W,D)
                img = torch.from_numpy(img).float().permute(1, 0, 2)

                #normalize image 0-1
                img = (img - img.min()) / (img.max() - img.min())


                seg_object_fix = self.reader.read(patient.replace('oct_', 'oct_seg_'))
                label, _ = self.reader.get_data(seg_object_fix)
                # map class labels
                label = remap_labels(label)

                label = torch.from_numpy(label).int().permute(1, 0, 2)


                coordinates = get_coords(label.shape)

                self.data_list.append({'pat_id': pat_id, 'lat': lat, 'pseudo_id' : self.pseudo_id, 'img' : img.squeeze(), 'label' : label, 'coordinates': coordinates.squeeze()})

            self.pseudo_id += 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # get item
        img = self.data_list[idx]['img']
        label = self.data_list[idx]['label']
        coordinates = self.data_list[idx]['coordinates']
        pat_id = self.data_list[idx]['pat_id']
        lat = self.data_list[idx]['lat']
        pseudo_id = self.data_list[idx]['pseudo_id']

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

        return {'img': image_patch, 'label': label_patch, 'coordinates': coords_patch, 'pat_id': pat_id, 'pseudo_id': pseudo_id, 'lat': lat}

