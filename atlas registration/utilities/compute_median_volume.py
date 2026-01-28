#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compute the median volume of a set of images.

"""

from utilities.files import read_yaml_config, compute_paths
from datasets.healthyOCT import HealthyOctDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot



FOLD = 0


if __name__ == '__main__':
    folds = read_yaml_config(" ")

    dataset = HealthyOctDataset(compute_paths(' ', folds['FOLD'][FOLD]['TRAIN'], subsmpl_factor=8 ), mode=None)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


    mean_pat = []
    mean_label = []
    for batch in dataloader:
        mean_pat.append(batch["img"])
        mean_label.append(batch["label"].unsqueeze(0).argmax(dim=-1))
    mean_pat, _ = torch.median(torch.cat(mean_pat, dim=0), dim=0)
    mean_label = torch.cat(mean_label).squeeze()
    mean_label, _ = torch.median(mean_label, dim=0)
    mean_label = one_hot(mean_label.type(torch.LongTensor), 10).squeeze().float()

    torch.save(mean_pat, f" ... /median_pat_oct_fold_{FOLD}.pt")
    torch.save(mean_label, f" ... /median_pat_label_fold_{FOLD}.pt")

