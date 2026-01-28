#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training routine to train INR model.

"""
import os
import torch
import lightning as L
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from utilities.metrics import sim_metrics


class GINRAtlasReg(L.LightningModule):

    def __init__(self, lr=1e-4, lr_exp_gamma=0.99, deformation_model=None, atlas_model=None, init_latent_code=None,
                 save_path=None):

        super().__init__()

        if deformation_model is None or atlas_model is None:
            raise RuntimeError("The models must be provided")

        if init_latent_code is None:
            raise RuntimeError("The init_latent_code must be provided")

        if atlas_model is None:
            raise RuntimeError("The init_atlas must be provided")

        self.atlas = atlas_model
        self.deformation_model = deformation_model

        self.lr = lr
        self.lr_exp_gamma = lr_exp_gamma
        self.latent_code = init_latent_code

        if save_path is not None:
            os.makedirs( save_path, exist_ok=True)
        self.save_path = save_path

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()

        self.save_hyperparameters(ignore=['atlas_model', 'deformation_model','latent_code' ])


    def training_step(self, batch, batch_idx):
        img = batch['img'].squeeze()
        label = batch['label'].squeeze()
        coords = batch['coordinates'].view(-1, 3)
        pseudo_id = batch['pseudo_id'].item()

        latent_code = self.latent_code[pseudo_id]

        #### Predictions
        displacement = self.deformation_model((coords, latent_code))

        deformation = displacement + coords

        recon, seg = self.atlas(deformation)

        #### Loss

        sim_los = self.mse_loss(recon, img.view(-1,1)) + self.bce_loss(seg, label.view(-1,label.shape[-1]))

        reg_loss = 0.01 * self.l1_loss(displacement, torch.zeros_like(displacement))

        loss = sim_los + reg_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('sim_loss', sim_los, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('reg_loss', reg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):

        if batch_idx != 10:
            return

        img = batch['img'].squeeze()
        label = batch['label'].squeeze()
        coords = batch['coordinates'].view(-1, 3)
        pseudo_id = batch['pseudo_id'].item()

        latent_code = self.latent_code[pseudo_id]

        #### Predictions
        displacement = self.deformation_model((coords, latent_code))

        deformation = displacement + coords

        recon, seg = self.atlas(deformation)

        #### Metrics

        seg = seg.argmax(dim=-1).squeeze().view(img.shape)
        label = label.argmax(dim=-1).squeeze()
        recon = recon.view(img.shape)

        # 2D evaluation
        metrics = sim_metrics(recon.permute(2, 0, 1), seg.permute(2, 0, 1), img.permute(2, 0, 1),
                              label.permute(2, 0, 1))

        dice = metrics['dice'][1:].mean()
        ssim = metrics['ssim'].item()
        assd = metrics['assd'].mean()

        self.log('val_dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_assd', assd, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return

    def test_step(self, batch, batch_idx):

        img = batch['img'].squeeze()
        label = batch['label'].squeeze()
        coords = batch['coordinates'].view(-1, 3)
        pseudo_id = batch['pseudo_id'].item()

        latent_code = self.latent_code[pseudo_id]

        #### Predictions
        displacement = self.deformation_model((coords, latent_code))

        deformation = displacement + coords

        recon, seg = self.atlas(deformation)

        #### Loss

        sim_los = self.mse_loss(recon, img.view(-1, 1)) + self.bce_loss(seg, label.view(-1, label.shape[-1]))

        reg_loss = 0.01 * self.l1_loss(displacement, torch.zeros_like(displacement))

        loss = sim_los + reg_loss

        #### Metrics


        seg = seg.argmax(dim=-1).squeeze().view(img.shape)
        label = label.argmax(dim=-1).squeeze()
        recon = recon.view(img.shape)

        # 2D evaluation
        metrics = sim_metrics(recon.permute(2,0,1), seg.permute(2,0,1), img.permute(2,0,1), label.permute(2,0,1))

        dice = metrics['dice'][1:].mean()
        ssim = metrics['ssim'].item()
        assd = metrics['assd'].mean()

        #### Visual

        plt.imshow(recon.view(img.shape)[..., 0].cpu().detach(), cmap='gray')
        if self.save_path is not None:
            file_name = f'test_recon_batchID_{batch_idx}_0.png'
            path = os.path.join(self.save_path, file_name)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

        plt.imshow(recon.view(img.shape)[..., img.shape[2]//2].cpu().detach(), cmap='gray')
        if self.save_path is not None:
            file_name = f'test_recon_batchID_{batch_idx}.png'
            path = os.path.join(self.save_path, file_name)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

        plt.imshow(seg.view(img.shape)[..., img.shape[2]//2].cpu().detach())
        if self.save_path is not None:
            file_name = f'test_label_batchID_{batch_idx}.png'
            path = os.path.join(self.save_path, file_name)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

        displacement = (displacement + 1) / 2

        plt.imshow(displacement.view(img.shape[0], img.shape[1], img.shape[2], 3)[..., img.shape[2]//2, :].cpu().detach())
        if self.save_path is not None:
            file_name = f'test_displacement_batchID_{batch_idx}.png'
            path = os.path.join(self.save_path, file_name)
            plt.savefig(path)
            plt.close()
        else:
            plt.show()

        plt.close()


        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('dice', dice, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('ssim', ssim, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('assd', assd, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.deformation_model.parameters(), 'lr': self.lr},
            {'params': self.atlas.parameters(), 'lr': 0.1 * self.lr},
            {'params': self.latent_code, 'lr': self.lr},
        ])
        scheduler = ExponentialLR(optimizer, gamma=self.lr_exp_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()

