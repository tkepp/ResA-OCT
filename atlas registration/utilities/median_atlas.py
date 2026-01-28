#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to pre-train atlas model.
"""
from scripts.architectures.reconSegMultiHead import ReconSegMuliHead
from scripts.architectures.SIREN import Siren
from scripts.utilities.files import read_yaml_config, compute_paths
import torch
from lightning import seed_everything
from torch.utils.data import DataLoader
from scripts.datasets.healthyOCT import HealthyOctDataset
from scripts.routines.intensityAtlas import IntensityAtlas
import lightning as L

DEST_PATH = ''

seed_everything(0, workers=True)
torch.set_float32_matmul_precision('medium')

config = read_yaml_config("./configs/fit_GINR_Atlas_base.yml")

folds = read_yaml_config("./configs/cross-validation-folds.yml")

paths = compute_paths(data_path=config['SETTINGS']['DATA_DIR'] ,patients_list=folds['FOLD'][config['SETTINGS']['FOLD']]['TRAIN'], subsmpl_factor=16)

dataset = HealthyOctDataset(paths, mode = 'downsample', patch_size=config['DATA']['PATCH_SIZE'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


atlas_model = Siren([config['ATLAS_MODEL']['IN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'],config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE']])
atlas_model = ReconSegMuliHead(atlas_model, config['ATLAS_MODEL']['HIDDEN_SIZE'], 10)

atlas = IntensityAtlas(model=atlas_model)

trainer = L.Trainer(accelerator='gpu', devices=[0], max_epochs=500, log_every_n_steps=1)

trainer.fit(atlas, train_dataloaders=dataloader)

torch.save(atlas.model.state_dict(), DEST_PATH)