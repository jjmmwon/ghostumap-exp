from ghostumap import GhostUMAP
from utils import tab10, ghost_plot, umap_embeddings, plot_embeddings


def load_celegans(
    path1="../data/celegans/celegans_proccessed.csv",
    path2="../data/celegans/celegans_metadata.csv",
):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(path1, index_col=0)
    df2 = pd.read_csv(path2, index_col=0)
    cell_types = df2["cell_type"]
    cell_types = cell_types.fillna("")

    ctype_to_lineage = {
        "": "Not annotated",
        ##
        "Neuroblast_ASE_ASJ_AUA": "ASE, ASJ, AUA",
        "Neuroblast_ASJ_AUA": "ASE, ASJ, AUA",
        "ASE_parent": "ASE, ASJ, AUA",
        "ASE": "ASE, ASJ, AUA",
        "ASEL": "ASE, ASJ, AUA",
        "ASER": "ASE, ASJ, AUA",
        "ASJ": "ASE, ASJ, AUA",
        "AUA": "ASE, ASJ, AUA",
        ##
        "Neuroblast_ASG_AWA": "ASG, AWA",
        "ASG_AWA": "ASG, AWA",
        "ASG": "ASG, AWA",
        "AWA": "ASG, AWA",
        ##
        "Neuroblast_ADF_AWB": "ADF, AWB",
        "ADF_AWB": "ADF, AWB",
        "ADF": "ADF, AWB",
        "AWB": "ADF, AWB",
        ##
        "AWC": "AWC",
        "AWC_ON": "AWC",
        ##
        "Neuroblast_AFD_RMD": "AFD",
        "AFD": "AFD",
        ##
        "ADL_parent": "ADL",
        "ADL": "ADL",
        ##
        "ASH": "ASH",
        ##
        "ASI_parent": "ASI",
        "ASI": "ASI",
        ##
        "ASK_parent": "ASK",
        "ASK": "ASK",
    }
    lineages = np.array([ctype_to_lineage[celltype] for celltype in cell_types])
    lineage_to_label = dict(
        zip(np.unique(lineages), list(range(len(np.unique(cell_types)))))
    )

    X = np.array(df)
    y = np.array([lineage_to_label[lineage] for lineage in lineages])
    names = np.array(list(lineage_to_label.keys()))
    colors = {
        "Not annotated": tab10[9],
        # blue
        "ASE, ASJ, AUA": tab10[0],
        # purple
        "ASG, AWA": tab10[4],
        # teal
        "ADF, AWB": tab10[8],
        # green
        "AWC": tab10[2],
        # yellow
        "AFD": tab10[7],
        # red
        "ADL": tab10[3],
        # orange
        "ASH": tab10[1],
        # pink
        "ASI": tab10[6],
        # brown
        "ASK": tab10[5],
    }

    return X, y, names, colors


X, y, names, colors = load_celegans()

n_neighbors = 15
min_dist = 0.1

(
    Zs,
    Zs_ghost,
    ghost_indices,
) = GhostUMAP(
    n_neighbors=n_neighbors, min_dist=min_dist
).fit_transform(X, n_embeddings=1, n_ghosts=4)

Z = Zs[0]
Z_ghost = Zs_ghost[0]

ghost_plot(Z, Z_ghost, y, names, colors, linewidth=0.5)

# comparison with multiple UMAP runs (with fixed knn and init)
Z_umaps = umap_embeddings(X, n_embeddings=2)
plot_embeddings(Z_umaps, y, names, colors)
