from ghostumap import GhostUMAP
from utils import tab10, ghost_plot, umap_embeddings, plot_embeddings

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale

dataset = load_breast_cancer()
X = scale(dataset.data)
y = dataset.target
names = dataset.target_names
colors = {"malignant": tab10[3], "benign": tab10[0]}

n_neighbors = 10
min_dist = 0.0
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
