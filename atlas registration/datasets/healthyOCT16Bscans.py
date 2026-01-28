#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset to load and prepare volumes in 8 B-Scan blocks.

"""

from torch.nn.functional import one_hot
from monai.data import NibabelReader
import torch
from torch.utils.data import Dataset
from utilities.Patches import  get_coords, remap_labels



class HealthyOctDataset8BScans(Dataset):

    def __init__(self, path_list:list =None):
        """
        Dataset for healthy oct patients in 8 B-Scan blocks.

        :param path_list: list of paths to images.
        """

        if path_list is None:
            raise RuntimeError("Paths are missing!")

        if not isinstance(path_list, list):
            raise RuntimeError("path_list is not a list!")

        self.reader = NibabelReader()

        self.pseudo_id = 0

        self.data_list = []

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

                img.squeeze()

                seg_object_fix = self.reader.read(patient.replace('oct_', 'oct_seg_'))
                label, _ = self.reader.get_data(seg_object_fix)
                # map class labels
                label = remap_labels(label)

                label = torch.from_numpy(label).int().permute(1, 0, 2)

                coordinates = get_coords(label.shape)

                num_sub_vol= img.shape[-1]//8

                for i in range(num_sub_vol):
                    sub_img = img[..., i*8:(i+1)*8]
                    sub_label = label[..., i*8:(i+1)*8]
                    sub_coordinates = coordinates[..., i*8:(i+1)*8, :]
                    self.data_list.append({'pat_id': pat_id, 'lat': lat, 'pseudo_id' : self.pseudo_id, 'img' : sub_img, 'label' : sub_label, 'coordinates': sub_coordinates})

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

        image_patch = img.squeeze()
        label_patch = label.squeeze().type(torch.LongTensor)
        label_patch = one_hot(label_patch, num_classes=10).float().squeeze()
        coords_patch = coordinates.squeeze()

        coords_patch.requires_grad = True

        return {'img': image_patch, 'label': label_patch, 'coordinates': coords_patch, 'pat_id': pat_id, 'pseudo_id': pseudo_id, 'lat': lat}

