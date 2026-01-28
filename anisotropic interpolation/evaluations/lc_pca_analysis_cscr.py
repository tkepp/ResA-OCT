import numpy as np
import os
from glob import glob
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = ''
RESULT_PATH = ''


if __name__ == '__main__':

    #configs for plot and colors
    tint_palette = [
        (180, 234, 221),
        (123, 221, 195),
        (103, 203, 175),
        (74, 185, 154),
        (58, 167, 134),
        (36, 149, 113),
        (3, 128, 102),
        (3, 128, 102),
        (0, 116, 98),
        (0, 116, 98),
        (0, 103, 91),
        (43, 203, 158)
    ]

    tint_palette = [tuple(channel / 256 for channel in color) for color in tint_palette]


    #TODO: choose colorpalatte
    #sns.set_palette(tint_palette)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Latin Modern Sans"],
        'font.size': 12
    })

    pat_list = []
    lc = []
    for fold_dir in sorted(glob(os.path.join(DATA_PATH, "LC_Fit_Patient*"))):

        pat_list.append(fold_dir.split("_")[-3])
        lc.append(torch.load(f'{fold_dir}/model_last.pt')['latent_codes'])

    model = torch.load(os.path.join(DATA_PATH, 'model_last.pt'), map_location='cpu')
    lcs_train = model['latent_codes']

    lc = torch.cat(lc, dim=0)

    # Run PCA
    pca = PCA(n_components=2, random_state=42)
    lc_pca = pca.fit_transform(lc.detach().cpu().numpy())


    lcs_full = torch.cat((lcs_train, lc), dim=0)
    pca = PCA(n_components=2, random_state=42)
    lc_pca_full = pca.fit_transform(lcs_full.detach().cpu().numpy())

    hue = np.zeros(len(lcs_full))
    hue[len(lcs_train):] = 1
    hue_label = ['test' if hue[i]==1 else 'train' for i in range(len(hue))]
    fig, ax = plt.subplots()
    sns.scatterplot(x=lc_pca_full[:, 0], y=lc_pca_full[:, 1], hue=hue_label, s=80)
    ax.title.set_text("CSCR Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_box_aspect(1)
    ax.set_xlim([-0.22, 0.48])
    ax.set_ylim([-0.28, 0.42])
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{RESULT_PATH}/pca_analysis_train_test_cscr.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()


    pca = PCA(n_components=2, random_state=42)
    lc_pca_train = pca.fit_transform(lcs_train.detach().cpu().numpy())

    hue = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           2,
           3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
           4, 4,
           5, 5, 5, 5,
           6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
           7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
           8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
           9, 9, 9, 9,
           10, 10, 10, 10, 10, 10, 10, 10,
           11,
           12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
           13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
           14, 14, 14, 14]

    hue_new = [str(item) for item in hue]

    fig, ax = plt.subplots()
    g = sns.scatterplot(x=lc_pca_train[:, 0], y=lc_pca_train[:, 1], hue=hue_new, s=80)
    g.legend().set_title('Patient ID')
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.title.set_text("CSCR Data: Training Only")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{RESULT_PATH}/pca_analysis_train_CSCR.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()