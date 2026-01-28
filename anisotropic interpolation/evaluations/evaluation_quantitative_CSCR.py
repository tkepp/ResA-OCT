#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import numbers, Font

# turn off all warnings
import warnings
warnings.filterwarnings("ignore")

from evaluation import calculate_segm_metrics_cscr, calculate_regr_metrics


def remap_labels(seg):
    out_seg = np.zeros_like(seg)
    out_seg[seg > 0] = 1  # Retina
    out_seg[seg > 1] = 2  # Pathologies
    return out_seg


subject_ids = ['0008', '0009', '0038', '0039', '0060', '0096', '0099', '0115', '0137', '0150', '0163', '0165', '0185', '0192', '0215', '0287', '0292', '0313', '0323']

gt_base_path = ""

result_dirs = ['',
               '',
               '',
              ]

idx_known = np.arange(0, 25, 2)
idx_interp = np.arange(1, 25, 2)

# iterate over all result dirs
for idx, result_dir in enumerate(result_dirs):
    print(idx, result_dir)

    # select inr or linear/reg dir by index
    if idx < 2:
        result_path = os.path.join("", result_dir)
    else:
        result_path = os.path.join("", result_dir)

    mses = []
    maes = []
    psnrs = []
    ssims = []
    lpipss = []
    dices_retina = []
    dices_patho = []
    assds_ilm = []
    hds95_ilm = []
    hds_ilm = []
    subject_names = []

    for subject_id in subject_ids:

        gt_paths = sorted(glob.glob(os.path.join(gt_base_path, 'Patient_' + subject_id, 'Patient_' + subject_id + '*' + '_oct_preprocessed.nii.gz'), recursive=True))

        for gt_path in gt_paths:


            date = gt_path.split(os.sep)[-1][13:22]
            subject_name = f'Patient_{subject_id}_{date}'
            print(subject_name)

            gt_seg_path = gt_path.replace('oct_preprocessed', 'oct_seg_preprocessed')

            try:
                # load interpolation result
                if 'registration' == result_dir:
                    recon_path = os.path.join(result_path, 'Patient_' + subject_id,
                                              f'{subject_name}_oct_interpolated_registration.nii.gz')
                    seg_path = recon_path.replace('oct_interpolated', 'oct_seg_interpolated')
                elif 'linear' == result_dir:
                    recon_path = os.path.join(result_path, 'Patient_' + subject_id,
                                              f'{subject_name}_oct_interpolated.nii.gz')
                    seg_path = recon_path.replace('oct_interpolated', 'oct_seg_interpolated')
                elif 'GenFit' in result_dir:
                    try:
                        recon_path = glob.glob(os.path.join(result_path, 'nifti', f'{subject_name[:-1]}_{subject_name[-1]}_vol.nii.gz'))[0]
                        seg_path = glob.glob(os.path.join(result_path, 'nifti', f'{subject_name[:-1]}_{subject_name[-1]}_vol_seg.nii.gz'))[0]
                    except:
                        recon_path = glob.glob(os.path.join(result_path, f'LC_Fit_{subject_name[:-1]}_{subject_name[-1]}',
                                                            'nifti', f'{subject_name[:-1]}_{subject_name[-1]}_vol.nii.gz'))[0]
                        seg_path = glob.glob(os.path.join(result_path, f'LC_Fit_{subject_name[:-1]}_{subject_name[-1]}',
                                                          'nifti', f'{subject_name[:-1]}_{subject_name[-1]}_vol_seg.nii.gz'))[0]
                else:
                    recon_path = glob.glob(os.path.join(result_path, f'{subject_name[:-1]}_{subject_name[-1]}', 'nifti',
                                                        f'{subject_name[:-1]}_{subject_name[-1]}_vol.nii.gz'))[0]
                    seg_path = glob.glob(os.path.join(result_path, f'{subject_name[:-1]}_{subject_name[-1]}', 'nifti',
                                                      f'{subject_name[:-1]}_{subject_name[-1]}_vol_seg.nii.gz'))[0]
            except:
                print('File not found!')
                continue

            sx, sy, sz = 6/512, 2/496, 6/25

            gt_vol = np.asarray(nib.load(gt_path).get_fdata()) / 255.
            gt_vol = gt_vol.transpose(1, 2, 0)

            gt_seg_vol = remap_labels(np.asarray(nib.load(gt_seg_path).get_fdata()))
            gt_seg_vol = gt_seg_vol.transpose(1, 2, 0)

            if subject_name[-1] == 'R':
                gt_vol = gt_vol[:, ::-1, :].copy()
                gt_seg_vol = gt_seg_vol[:, ::-1, :].copy()

            pred_vol = np.asarray(nib.load(recon_path).get_fdata())

            pred_seg_vol = remap_labels(np.asarray(nib.load(seg_path).get_fdata()))

            if pred_vol.shape == (496, 230, 25):
                pred_vol = pred_vol.transpose(1, 0, 2)
                pred_seg_vol = pred_seg_vol.transpose(1, 0, 2)
            elif pred_vol.shape == (25, 230, 496):
                pred_vol = pred_vol.transpose(1, 2, 0)
                pred_seg_vol = pred_seg_vol.transpose(1, 2, 0)

            if idx < 2:
                pred_vol /= 255.

                if subject_name[-1] == 'R':
                    pred_vol = pred_vol[:, ::-1, :].copy()
                    pred_seg_vol = pred_seg_vol[:, ::-1, :].copy()

            #print(gt_vol.shape, pred_vol.shape,
            #      gt_vol.min(), gt_vol.max(), pred_vol.min(), pred_vol.max(),
            #      gt_seg_vol.min(), gt_seg_vol.max(), pred_seg_vol.min(), pred_seg_vol.max())

            metrics_recon = calculate_regr_metrics(gt_vol[..., idx_interp], pred_vol[..., idx_interp])
            mses.append(metrics_recon['MSE'] * 100)
            maes.append(metrics_recon['MAE'] * 100)
            psnrs.append(metrics_recon['PSNR'])
            ssims.append(metrics_recon['SSIM'] * 100)
            lpipss.append(metrics_recon['LPIPS'])

            metrics_seg = calculate_segm_metrics_cscr(gt_seg_vol[..., idx_interp],
                                                      pred_seg_vol[..., idx_interp],
                                                      sampling=np.array([sx, sy]))

            dices_retina.append(metrics_seg['DICE RETINA'] * 100) # save in %
            dices_patho.append(metrics_seg['DICE FLUIDS'] * 100)  # save in %
            assds_ilm.append(metrics_seg['ASSD ILM'] * 1000) # save in µm
            hds95_ilm.append(metrics_seg['HD 95 ILM'] * 1000) # save in µm
            hds_ilm.append(metrics_seg['HD ILM'] * 1000) # save in µm

            subject_names.append(subject_name)
            print('\n')

    # Create DataFrame
    df = pd.DataFrame(
        {'Subject Name': subject_names, 'MSE': mses, 'MAE': maes, 'PSNR': psnrs, 'SSIM': ssims, 'LPIPS': lpipss,
         'Dice Retina': dices_retina, 'Dice Fluid': dices_patho,
         'ASSD ILM': assds_ilm, 'HD95 ILM': hds95_ilm, 'HD ILM': hds_ilm})


    out_xls_path = f"results_{result_dir}.xlsx"

    # Save the DataFrame to an Excel file without formatting
    df.to_excel(out_xls_path, index=False)