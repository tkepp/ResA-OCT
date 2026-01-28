#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import glob

# i/o paths
csv_path = ''

# read csv study data
df = pd.read_csv(csv_path)
subject_ids = df['subject'].to_list()
lateralities = df['eye'].to_list()

# custom colormap for segmentations
custom_colormap = ListedColormap([
    (0, 0, 0),       # Klasse 0 - Schwarz
    (0, 0.4, 0.4),   # Klasse 1 - Dunkeltürkis
    (0, 0.2, 0.3),   # Klasse 2 - Petrol
    (0.2, 0.4, 0.5), # Klasse 3 - Graupetrol
    (0.6, 0.2, 0.4), # Klasse 4 - Dunkelpink
    (0.8, 0.4, 0.6), # Klasse 5 - Pink
    (1, 0.6, 0.8),   # Klasse 6 - Helles Pink
    (0.9, 0.7, 0.8), # Klasse 7 - Rosé
    (0.7, 0.5, 0.6), # Klasse 8 - Altrosa
    (0.5, 0.3, 0.4), # Klasse 9 - Violettgrau
    (0.3, 0.1, 0.2), # Klasse 10 - Tiefviolett
    (0.1, 0.05, 0.1),# Klasse 11 - Dunkelviolett
    (0, 0, 0)        # Klasse 12 - Schwarz
])

# line width for frames
mpl.rcParams['axes.linewidth'] = 1.5

for idx, (subject_id,eye) in enumerate(zip(subject_ids,lateralities)):

    # perform both plotting recon and seg results
    for img_label_type in ['', '_seg']:

        # load all image paths for each patients
        reference_paths = sorted(glob.glob(f''))

        reg_interp_paths = sorted(glob.glob(f''))

        lin_interp_paths = sorted(glob.glob(f''))

        singleINR_paths = sorted(glob.glob(f''))

        singleINR_SLO_paths = sorted(glob.glob(f''))

        genINR_paths = sorted(glob.glob(f'', recursive=True))

        genINR_SLO_paths = sorted(glob.glob(f'', recursive=True))

        # define settings for image and segmentation plots
        if img_label_type == '':
            params = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
        else:
            params = {'cmap': custom_colormap, 'interpolation': 'none'}

        # plot for selection of interpolated slices
        for interp_slice in [1, 13, 15, 17]:

            if not interp_slice % 2:
                print('Interpolated slices need to be in [1, 3, ..., 29]!')
                sys.exit(1)

            slice1 = interp_slice - 1
            slice2 = interp_slice + 1

            # load image/label data
            orig_ref = np.asarray(nib.load(reference_paths[idx]).get_fdata())
            if img_label_type == '':
                orig_ref /= 255.
            orig_slice1 = orig_ref[..., slice1]
            orig_slice2 = orig_ref[..., interp_slice]
            orig_slice3 = orig_ref[..., slice2]

            interp_linear = np.asarray(nib.load(lin_interp_paths[idx]).get_fdata())[..., interp_slice]
            interp_reg = np.asarray(nib.load(reg_interp_paths[idx]).get_fdata())[..., interp_slice]
            interp_singleINR = np.asarray(nib.load(singleINR_paths[idx]).get_fdata())[..., interp_slice]
            interp_singleINR_SLO = np.asarray(nib.load(singleINR_SLO_paths[idx]).get_fdata())[..., interp_slice]
            interp_genINR = np.asarray(nib.load(genINR_paths[idx]).get_fdata())[..., interp_slice]
            interp_genINR_SLO = np.asarray(nib.load(genINR_SLO_paths[idx]).get_fdata())[..., interp_slice]

            if img_label_type == '':
                interp_linear /= 255.
                interp_reg /= 255.


            for (img_label, name) in zip([orig_slice1.T, orig_slice2.T, orig_slice3.T, interp_linear.T, interp_reg.T, interp_singleINR.T,
                                  interp_singleINR_SLO.T,interp_genINR.T,interp_genINR_SLO.T],
                                 ['InSlice1','OrigSliceIntermed', 'InSlice2','Linear','SingleINR','GenINR',
                                  'Registration','SingleINR_SLO','GenINR_SLO']):

                # create output dir
                output_dir = os.path.join(f'')
                os.makedirs(output_dir, exist_ok=True)

                # plot single images without text overlays
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.imshow(img_label, **params)
                plt.subplots_adjust(left=0.0,
                                    bottom=0.00,
                                    right=1.0,
                                    top=1,
                                    wspace=0,
                                    hspace=0)
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'{subject_id:02d}_slice{interp_slice:02d}_{eye}_{name}{img_label_type}.png'), dpi=512, bbox_inches='tight', pad_inches=0)
                plt.close()

            # select typewriter font --> DFKI corporate design
            font_path = ""  # Falls sie installiert ist
            font_prop = fm.FontProperties(fname=font_path)

            # plot summary plot
            fig, axes = plt.subplots(3, 3, figsize=(6, 3))
            axes[0, 0].imshow(orig_slice1.T, **params)
            axes[0, 0].text(10, 35, 'Input Slice 1', color='white', size='smaller', fontproperties=font_prop)
            axes[0, 1].imshow(orig_slice2.T, **params)
            axes[0, 1].text(10, 35, 'Original Intermediate Slice', color='white', size='smaller', fontproperties=font_prop)
            axes[0, 2].imshow(orig_slice3.T, **params)
            axes[0, 2].text(10, 35, 'Input Slice 2', color='white', size='smaller', fontproperties=font_prop)
            #
            axes[1, 0].imshow(interp_linear.T, **params)
            axes[1, 0].text(10, 35, 'Linear', color='white', size='smaller', fontproperties=font_prop)
            axes[2, 0].imshow(interp_reg.T, **params)
            axes[2, 0].text(10, 35, 'Registration-based', color='white', size='smaller', fontproperties=font_prop)
            #
            axes[1, 1].imshow(interp_singleINR.T, **params)
            axes[1, 1].text(10, 35, 'SingleINR', color='white', size='smaller', fontproperties=font_prop)
            axes[2, 1].imshow(interp_singleINR_SLO.T, **params)
            axes[2, 1].text(10, 35, '$\mathregular{SingleINR_{SLO}}$', color='white', size='smaller', fontproperties=font_prop)
            #
            axes[1, 2].imshow(interp_genINR.T, **params)
            axes[1, 2].text(10, 35, 'GenINR', color='white', size='smaller', fontproperties=font_prop)
            axes[2, 2].imshow(interp_genINR_SLO.T, **params)
            axes[2, 2].text(10, 35, '$\mathregular{GenINR_{SLO}}$', color='white', size='smaller', fontproperties=font_prop)
            #
            # adjust spaces
            plt.subplots_adjust(left=0.0,
                                bottom=0.03,
                                right=1.0,
                                top=0.99,
                                wspace=0,
                                hspace=0.12)

            # set frame colors
            for i in range(3):
                for j in range(3):
                    axes[i, j].spines['top'].set_color((129 / 255, 233 / 255, 202 / 255))
                    axes[i, j].spines['left'].set_color((129 / 255, 233 / 255, 202 / 255))
                    axes[i, j].spines['right'].set_color((129 / 255, 233 / 255, 202 / 255))
                    axes[i, j].spines['bottom'].set_color((129 / 255, 233 / 255, 202 / 255))

            axes[0, 0].spines['top'].set_color((253 / 255, 138 / 255, 234 / 255))
            axes[0, 0].spines['left'].set_color((253 / 255, 138 / 255, 234 / 255))
            axes[0, 0].spines['right'].set_color((253 / 255, 138 / 255, 234 / 255))
            axes[0, 0].spines['bottom'].set_color((253 / 255, 138 / 255, 234 / 255))

            axes[0, 2].spines['top'].set_color((253 / 255, 138 / 255, 234 / 255))
            axes[0, 2].spines['left'].set_color((253 / 255, 138 / 255, 234 / 255))
            axes[0, 2].spines['right'].set_color((253 / 255, 138 / 255, 234 / 255))
            axes[0, 2].spines['bottom'].set_color((253 / 255, 138 / 255, 234 / 255))

            # remove ticks
            [axi.set_xticks([]) for axi in axes.ravel()]
            [axi.set_yticks([]) for axi in axes.ravel()]

            # save figure
            plt.savefig(os.path.join('', f'{subject_id:02d}_{eye}_Slice_{interp_slice:02d}{img_label_type}.png'), dpi=512, transparent=True)

            plt.close()