import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale

from umap import UMAP

tab10 = {
    0: "#4E79A7",  # blue
    1: "#F28E2B",  # orange
    2: "#59A14F",  # green
    3: "#E15759",  # red
    4: "#B07AA1",  # purple
    5: "#9C755F",  # brown
    6: "#FF9DA7",  # pink
    7: "#EDC948",  # yellow-ish
    8: "#76B7B2",  # teal
    9: "#BAB0AC",  # gray
}

# example 1. breast cancer
dataset = load_breast_cancer()
X = scale(dataset.data)
y = dataset.target
names = dataset.target_names
colors = {"malignant": tab10[3], "benign": tab10[0]}
cmap = LinearSegmentedColormap.from_list(
    "", [colors[names[label]] for label in np.unique(y)]
)

umap = UMAP(n_neighbors=10, min_dist=0.0)
# umap._input_hash = None
Z = umap.fit_transform(X)
# np.save("breast_cancer_Z.npy", Z)

_, ax = plt.subplots(figsize=(6, 6))
plt.scatter(*Z.T, c=y, cmap=cmap, linewidth=0.5, edgecolor="#aaaaaa", s=30)
# idx = (y == 1) & (Z[:, 0] > 2.8)
# plt.scatter(*Z[idx].T, color=colors[1], linewidth=2.0, edgecolor="#aaffaa")
ax.set_aspect("equal")
ax.set_box_aspect(1)
ax.axis("off")
plt.tight_layout()
plt.show()


# example 2. C. elegans
def load_celegans(
    path1="./data/celegans/celegans_proccessed.csv",
    path2="./data/celegans/celegans_metadata.csv",
):
    df = pd.read_csv(path1, index_col=0)
    df2 = pd.read_csv(path2, index_col=0)
    cell_types = df2["cell_type"]
    cell_types = cell_types.fillna("")
    ctype_to_label = dict(
        zip(np.unique(cell_types), list(range(len(np.unique(cell_types)))))
    )

    X = np.array(df)
    y = np.array([ctype_to_label[celltype] for celltype in cell_types])

    names = np.unique(cell_types)
    colors = {
        "": tab10[9],
        # blue
        "Neuroblast_ASE_ASJ_AUA": tab10[0],
        "Neuroblast_ASJ_AUA": tab10[0],
        "ASE_parent": tab10[0],
        "ASE": tab10[0],
        "ASEL": tab10[0],
        "ASER": tab10[0],
        "ASJ": tab10[0],
        "AUA": tab10[0],
        # purple
        "Neuroblast_ASG_AWA": tab10[4],
        "ASG_AWA": tab10[4],
        "ASG": tab10[4],
        "AWA": tab10[4],
        # teal
        "Neuroblast_ADF_AWB": tab10[8],
        "ADF_AWB": tab10[8],
        "ADF": tab10[8],
        "AWB": tab10[8],
        # green
        "AWC": tab10[2],
        "AWC_ON": tab10[2],
        # yellow
        "Neuroblast_AFD_RMD": tab10[7],
        "AFD": tab10[7],
        # red
        "ADL_parent": tab10[3],
        "ADL": tab10[3],
        # orange
        "ASH": tab10[1],
        # pink
        "ASI_parent": tab10[6],
        "ASI": tab10[6],
        # brown
        "ASK_parent": tab10[5],
        "ASK": tab10[5],
    }

    return X, y, names, colors


X, y, names, colors = load_celegans()
cmap = LinearSegmentedColormap.from_list(
    "", [colors[names[label]] for label in np.unique(y)]
)

umap = UMAP(n_neighbors=15, min_dist=0.1)
Z = umap.fit_transform(X)
# np.save("celegans_Z.npy", Z)

_, ax = plt.subplots(figsize=(6, 6))
plt.scatter(*Z.T, c=y, cmap=cmap, linewidth=0.3, edgecolor="#aaaaaa", s=15)
ax.set_aspect("equal")
ax.set_box_aspect(1)
ax.axis("off")
plt.tight_layout()
plt.show()
