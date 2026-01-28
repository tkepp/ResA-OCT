#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reconstruction and segmentation head for MLP to predict intensity and label outputs.
"""
import torch.nn as nn
import torch.nn.functional as F


class ReconSegMuliHead(nn.Module):
    def __init__(self, MLP, channels = 128, num_classes = 2):
        super(ReconSegMuliHead, self).__init__()
        self.MLP = MLP
        self.intensity = nn.Linear(channels, 1)
        self.classifier = nn.Linear(channels, num_classes)


    def forward(self, x):
        x = self.MLP(x)

        # reconstruction head
        i = self.intensity(x)
        i = F.sigmoid(i)

        # segmentation head
        c = self.classifier(x)
        c = F.softmax(c, dim=-1)
        return i,c