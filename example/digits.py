from ghostumap import GhostUMAP
from utils import tab10, ghost_plot, umap_embeddings, plot_embeddings

from sklearn.datasets import load_digits

dataset = load_digits()
X = dataset.data
y = dataset.target
names = dataset.target_names
colors = {}
for i, name in enumerate(names):
    colors[name] = tab10[i]

(
    Zs,
    Zs_ghost,
    ghost_indices,
) = GhostUMAP().fit_transform(X, n_embeddings=1, n_ghosts=4)

Z = Zs[0]
Z_ghost = Zs_ghost[0]
ghost_plot(Z, Z_ghost, y, names, colors)

# comparison with multiple UMAP runs (with fixed knn and init)
Z_umaps = umap_embeddings(X, n_embeddings=2)
plot_embeddings(Z_umaps, y, names, colors)
