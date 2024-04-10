import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from umap import UMAP
from umap.umap_ import nearest_neighbors

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


def ghost_plot(
    Z, Z_ghost, y, names, colors, s=20, s_ghost=5, linewidth=1, show_legend=True
):
    n_insts = Z.shape[0]
    n_attrs = Z.shape[1]
    n_ghosts = Z_ghost.shape[1]
    n_ghost_insts = n_insts * n_ghosts

    Z_ghost_flatten = Z_ghost.reshape(n_ghost_insts, n_attrs)
    y_ghost = y.repeat(n_ghosts)

    source_positions = np.tile(Z, (1, n_ghosts)).reshape((n_ghost_insts, n_attrs))
    target_positions = Z_ghost_flatten
    line_positions = np.hstack((source_positions, target_positions)).reshape(
        (n_ghost_insts, 2, n_attrs)
    )

    total_moving_dists = (
        np.linalg.norm(source_positions - target_positions, axis=1)
        .reshape((n_insts, n_ghosts))
        .sum(axis=1)
    )
    total_moving_dists /= total_moving_dists.max()

    cmap = LinearSegmentedColormap.from_list(
        "", [colors[names[label]] for label in np.unique(y)]
    )
    plot_order = np.argsort(total_moving_dists)
    plot_order_ghost = plot_order.repeat(n_ghosts) * n_ghosts
    for i in range(n_ghosts):
        plot_order_ghost[i::n_ghosts] += i

    _, ax = plt.subplots(figsize=(6, 6))
    line_collection = LineCollection(
        line_positions,
        color="#000000",
        alpha=np.repeat(total_moving_dists, n_ghosts),
        linewidth=linewidth,
        antialiaseds=(1,),
    )
    line_collection.set_zorder(3)
    ax.add_collection(line_collection)

    ax.scatter(*Z[plot_order].T, c=y[plot_order], cmap=cmap, s=s)
    ax.scatter(
        *Z_ghost_flatten[plot_order_ghost].T,
        c=y_ghost[plot_order_ghost],
        cmap=cmap,
        s=s_ghost,
        alpha=np.repeat(total_moving_dists, n_ghosts)[plot_order_ghost]
    )

    if show_legend:
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markersize=7,
                    label=names[label],
                    markerfacecolor=colors[names[label]],
                )
                for label in np.unique(y)
            ],
        )
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def fixed_init_knn(X, n_neighbors=15, min_dist=0.1):
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_epochs=0)
    init = umap.fit_transform(X)
    precomputed_knn = nearest_neighbors(
        X,
        n_neighbors=n_neighbors,
        metric=umap.metric,
        metric_kwds=umap.metric_kwds,
        angular=umap.angular_rp_forest,
        random_state=umap.random_state,
    )
    return init, precomputed_knn


def umap_embeddings(X, n_neighbors=15, min_dist=0.1, n_embeddings=2):
    # generate multiple umap embs with fixed init and knn
    init, precomputed_knn = fixed_init_knn(X, n_neighbors, min_dist)
    umap = UMAP(init=init, precomputed_knn=precomputed_knn)
    embeddings = []
    for _ in range(n_embeddings):
        embeddings.append(umap.fit_transform(X))
    return embeddings


def plot_embeddings(embeddings, y, names, colors, s=15):
    cmap = LinearSegmentedColormap.from_list(
        "", [colors[names[label]] for label in np.unique(y)]
    )
    _, axs = plt.subplots(
        nrows=1,
        ncols=len(embeddings),
        figsize=(5 * len(embeddings), 5),
        tight_layout=True,
    )

    for i, Z in enumerate(embeddings):
        axs[i].scatter(*Z.T, c=y, cmap=cmap, s=s)
        axs[i].set_aspect("equal")
        axs[i].set_box_aspect(1)
    plt.tight_layout()
    plt.show()
