import os
from glob import glob
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = ""
RESULT_PATH = ""


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


    lc = torch.cat(lc, dim=0)

    # Run PCA
    pca = PCA(n_components=2, random_state=42)
    lc_pca = pca.fit_transform(lc.detach().cpu().numpy())

    # Plot
    sns.scatterplot(x=lc_pca[:, 0], y=lc_pca[:, 1], hue=pat_list)
    plt.title("PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(f"{RESULT_PATH}/pca_analysis.pdf", bbox_inches='tight', pad_inches=0.1)
    plt.show()

