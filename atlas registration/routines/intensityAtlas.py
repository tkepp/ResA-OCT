#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training routine for SIREN for image representation

"""

import torch
import lightning as L
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from utilities.Patches import get_coords


class IntensityAtlas(L.LightningModule):

    def __init__(self, lr=1e-4, model=None, result_dir :str =None):

        super().__init__()

        if model is None:
            raise RuntimeError("The model must be provided")

        self.result_dir = result_dir
        self.lr = lr
        self.model = model

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.save_hyperparameters(ignore=['model'])


    def training_step(self, batch, batch_idx):
        img = batch['img'].squeeze()
        label = batch['label'].squeeze()
        coords = batch['coordinates'].view(-1, 3)

        recon, seg = self.model(coords)

        loss = self.mse_loss(recon, img.view(-1,1)) + self.bce_loss(seg, label.view(-1,label.shape[-1]))

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        coords = get_coords((231,496,3)).to(batch['img'])

        recon, seg = self.model(coords)

        seg = seg.argmax(dim=-1)

        plt.imshow(recon.view(231,496,2)[..., 0].cpu().detach(), cmap='gray')
        plt.show()

        plt.imshow(seg.view(231, 496, 2)[..., 0].cpu().detach())
        plt.show()

        plt.close()

        return



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ExponentialLR(optimizer, gamma=0.99)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()
