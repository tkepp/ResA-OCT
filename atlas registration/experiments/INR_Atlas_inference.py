#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference for singe INR.

"""
import os
import time
from datetime import datetime
import torch
import yaml
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
import lightning as L
from argparse import ArgumentParser

from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb

from architectures.SIREN import Siren
from architectures.reconSegMultiHead import ReconSegMuliHead
from datasets.healthyOCT import HealthyOctDataset
from datasets.healthyOCT16Bscans import HealthyOctDataset16BScans
from routines.INR_Atlas_Reg_Inference import INRAtlasRegInference
from utilities.files import read_yaml_config, compute_paths
from utilities.metrics import deformation_field_metrics, sim_metrics
import numpy as np


def mean_dicts(dict_list):
    if not dict_list:
        return {}

    result = {}
    keys = dict_list[0].keys()

    for key in keys:
        values = [d[key] for d in dict_list]

        # If values are lists, compute element-wise mean
        if isinstance(values[0], list):
            stacked = np.stack(values)
            result[key] = stacked.mean(axis=0).tolist()
        else:
            # Scalar mean
            result[key] = float(np.mean(values))

    return result

def manual_test(batch, model, atlas_only):
    img = batch['img'].squeeze()
    label = batch['label'].squeeze()
    coords = batch['coordinates'].view(-1, 3)
    pseudo_id = batch['pseudo_id'].item()

    #### Predictions
    if atlas_only:
        deformation = coords.to()
    else:
        displacement = torch.sin(model.deformation_model(coords))

        deformation = displacement + coords

    recon, seg = model.atlas(deformation)

    seg = seg.argmax(dim=-1)
    label = label.argmax(dim=-1)

    recon = recon.view(img.shape)
    seg = seg.view(img.shape)

    metrics = sim_metrics(recon.squeeze().permute(2, 0, 1),
                          seg.permute(2, 0, 1),
                          img.squeeze().permute(2, 0, 1),
                          label.squeeze().permute(2, 0, 1))

    deformation_metric = deformation_field_metrics(deformation, coords)

    masked_deformation_metric = deformation_field_metrics(deformation * (label > 0).repeat(1,1,1,3).view(-1, 3), coords)

    metrics["neg_jacobian_metric"] = deformation_metric['neg_jac_det_perc']

    metrics["masked_neg_jacobian_metric"] = masked_deformation_metric['neg_jac_det_perc']

    if atlas_only:
        metrics["l1_deformation"] = torch.tensor(0.)
    else:
        metrics["l1_deformation"] = torch.nn.functional.l1_loss(displacement, torch.zeros_like(displacement))


    return metrics


def main(hparams):

    # set seeds for all random stuff
    seed_everything(0, workers=True)
    torch.set_float32_matmul_precision('medium')

    config = read_yaml_config(hparams.config)

    folds = read_yaml_config("./configs/cross-validation-folds.yml")

    # override config
    if hparams.epochs is not None:
        config['TEST']["TOTAL_EPOCHS"] = hparams.epochs

    if hparams.lr is not None:
        config['TEST']["LEARNING_RATE"] = hparams.lr


    # set project
    if not hparams.justAtlas:
        project = f"{config['SETTINGS']['PROJECT_NAME']}_evaluation"
    else:
        project = f"{config['SETTINGS']['PROJECT_NAME']}_evaluation_atlas_only"

    # define params
    batch_size = config['TEST']["BATCH_SIZE"]
    lr = config['TEST']["LEARNING_RATE"]
    epochs = config['TEST']["TOTAL_EPOCHS"]


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


    paths = compute_paths(data_path=config['SETTINGS']['DATA_DIR'] ,patients_list=folds['FOLD'][config['SETTINGS']['FOLD']]['TEST'], subsmpl_factor=config['DATA']["SUBSMPL_FACTOR"])

    atlas_model = Siren(
        [config['ATLAS_MODEL']['IN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE'],
         config['ATLAS_MODEL']['HIDDEN_SIZE'], config['ATLAS_MODEL']['HIDDEN_SIZE']])
    atlas_model = ReconSegMuliHead(atlas_model, config['ATLAS_MODEL']['HIDDEN_SIZE'], 10)

    atlas_model.load_state_dict(
        torch.load(config['ATLAS_MODEL']['PATH'], weights_only=True))

    for pat in paths:

        patient = pat.split('/')[-3]
        lat =  pat.split('/')[-2]

        sub_run = run + patient + lat


        wandb_logger = WandbLogger(project=project, save_dir=result_path, name=sub_run)


        #load data sampled for training
        dataset = HealthyOctDataset([pat], mode = config['DATA']['SAMPLE_MODE'], patch_size=config['DATA']['PATCH_SIZE'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


        displacement_model =  Siren([3,128,128,128,128,3], omega=config['DISP_MODEL']['OMEGA'], init_factor=0.001)



        model = INRAtlasRegInference(
            lr=lr,
            lr_exp_gamma = config['TEST']["LR_SCHEDULER_GAMMA"],
            atlas_model=atlas_model,
            deformation_model=displacement_model,
            save_path=os.path.join(result_path, 'images', patient + lat)
        )

        if not hparams.justAtlas:

            # define callbacks
            lr_monitor = LearningRateMonitor(logging_interval='epoch')

            callbacks = [lr_monitor]

            #train model
            trainer = L.Trainer(accelerator='gpu', devices=[hparams.device], default_root_dir=result_path, max_epochs=epochs,  callbacks=callbacks, log_every_n_steps=1, logger = [wandb_logger], check_val_every_n_epoch=10)

            # Start time
            start_time = time.time()
            trainer.fit(model, train_dataloaders=dataloader)
            end_time = time.time()
            training_time = end_time - start_time


            dataset.patch_size = (231, 496, 16)
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            trainer.test(model, dataloaders=test_loader)

        else:
            model.cpu()
            training_time = 0

        dataset = HealthyOctDataset16BScans([pat])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        res = []
        for batch in dataloader:
            metrics = dict(manual_test(batch, model, atlas_only= hparams.justAtlas))
            print(metrics)
            metrics = {key: value.tolist() for key, value in metrics.items()}
            res.append(metrics)

        mean_dict = mean_dicts(res)

        print(mean_dict)

        results = mean_dict

        results["training_time"] = training_time


        # store results
        file = os.path.join(result_path, f'results_{patient + lat}.yaml')
        if not os.path.exists(file):
            # dump settings to config file
            with open(file, 'x') as file:
                yaml.safe_dump(results, file)
        else:
            with open(file, 'a') as file:
                yaml.safe_dump(results, file)


        wandb.finish()

        dataloader = None
        dataset = None
        del dataloader
        del dataset



if __name__ == "__main__":
    parser = ArgumentParser(description='INR for atlas registration')
    parser.add_argument('config', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument("--device",             default=0, type=int, help="The index of the GPU to use.")
    parser.add_argument("--epochs",             default=None, type=int, help="The number of epochs to train.")
    parser.add_argument("--lr",                 default=None, type=float, help="The initial learning rate.")
    parser.add_argument("--justAtlas", default=0, type=int, choices=[0,1] ,help="Evaluate just the atlas without deformation.")
    args = parser.parse_args()

    main(args)