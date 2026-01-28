#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to run training.

"""
import os
from datetime import datetime
import torch
import yaml
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader
from architectures.SIREN import Siren
from architectures.modulatedSIREN import ModulatedSiren
from architectures.reconSegMultiHead import ReconSegMuliHead
from datasets.healthyOCT import HealthyOctDataset
from routines.GINR_Atlas_Reg import GINRAtlasReg
from utilities.files import read_yaml_config, compute_paths


def main(hparams):

    # set seeds for all random stuff
    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision('medium')

    config = read_yaml_config(hparams.config)

    folds = read_yaml_config("./configs/cross-validation-folds.yml")

    # override config
    if hparams.epochs is not None:
        config['TRAINING']["TOTAL_EPOCHS"] = hparams.epochs

    if hparams.lr is not None:
        config['TRAINING']["LEARNING_RATE"] = hparams.lr


    # set project
    project = config['SETTINGS']["PROJECT_NAME"]


    # define params
    batch_size = config['TRAINING']["BATCH_SIZE"]
    lr = config['TRAINING']["LEARNING_RATE"]
    epochs = config['TRAINING']["TOTAL_EPOCHS"]


    # define run name
    run = "run_{}_FOLD_{}".format(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"), config['SETTINGS']['FOLD'])

    # set result dir
    result_path = os.path.join(config['SETTINGS']["RESULT_DIR"], '{}/{}'.format(project, run))
    os.makedirs(result_path, exist_ok=True)

    # dump settings to config file
    with open(os.path.join(result_path,'config.yaml'), 'x') as file:
        yaml.safe_dump(config, file)


    print("=================================")
    print(f"Running for project {project}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Fold: {config['SETTINGS']['FOLD']}")
    print(f"Should be running on device: {hparams.device}")
    print("==================================")


    wandb_logger = WandbLogger(project=project, save_dir=result_path, name=run)


    paths = compute_paths(data_path=config['SETTINGS']['DATA_DIR'] ,patients_list=folds['FOLD'][config['SETTINGS']['FOLD']]['TRAIN'], subsmpl_factor=config['DATA']["SUBSMPL_FACTOR"])

    print("Number of patients: {}".format(len(paths)))

    #load data sampled for training
    dataset = HealthyOctDataset(paths, mode = config['DATA']['SAMPLE_MODE'], patch_size=config['DATA']['PATCH_SIZE'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


    # initialize latent code
    latent_code_init = torch.cat([torch.normal(1., config['TRAINING']['LATENT_CODE']['INIT_STD'], (
    len(paths), config['TRAINING']['LATENT_CODE']['SIZE'] // 2)),
                                  torch.normal(0., config['TRAINING']['LATENT_CODE']['INIT_STD'], (
                                  len(paths), config['TRAINING']['LATENT_CODE']['SIZE'] // 2))], dim=-1)
    latent_code_init = torch.nn.Parameter(latent_code_init, requires_grad=True)

    # load pretrained atlas models
    atlas_model = Siren([config['ATLAS_MODEL']['IN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE']])
    atlas_model = ReconSegMuliHead(atlas_model, config['ATLAS_MODEL']['HIDDEN_SIZE'], 10)
    atlas_model.load_state_dict(
        torch.load(config['ATLAS_MODEL']['PATH'], weights_only=True))

    # initialize displacement model
    displacement_model =  ModulatedSiren(coord_size=config['DISP_MODEL']['IN_SIZE'], embed_size=config['TRAINING']['LATENT_CODE']['SIZE'], hidden_size= config['DISP_MODEL']['HIDDEN_SIZE'], num_hidden_layers=config['DISP_MODEL']['NUM_LAYERS'], output_size=config['DISP_MODEL']['OUT_SIZE'], siren_omega_0=config['DISP_MODEL']['OMEGA'], siren_init=0.001)


    model = GINRAtlasReg(
        lr = config['TRAINING']['LEARNING_RATE'],
        lr_exp_gamma=config['TRAINING']['LR_SCHEDULER_GAMMA'],
        deformation_model= displacement_model,
        atlas_model= atlas_model,
        init_latent_code = latent_code_init,
        save_path = os.path.join(result_path, 'images')
    )


    # define callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [lr_monitor]

    #train model
    trainer = L.Trainer(accelerator='gpu', devices=[hparams.device, hparams.device+1],  strategy=DDPStrategy(find_unused_parameters=True), default_root_dir=result_path, max_epochs=config['TRAINING']['TOTAL_EPOCHS'],  callbacks=callbacks, log_every_n_steps=1, logger = [wandb_logger], check_val_every_n_epoch=10)
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    dataset.patch_size= (231,496,16)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    trainer.test(model, dataloaders=test_loader)



if __name__ == "__main__":
    parser = ArgumentParser(description='INR for atlas registration')
    parser.add_argument('config', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument("--device",             default=0, type=int, help="The index of the GPU to use.")
    parser.add_argument("--epochs",             default=None, type=int, help="The number of epochs to train.")
    parser.add_argument("--lr",                 default=None, type=float, help="The initial learning rate.")
    args = parser.parse_args()

    main(args)