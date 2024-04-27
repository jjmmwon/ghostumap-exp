import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from umap import UMAP

from ghostumap import GhostUMAP
from ghostumap.utils import detect_instable_ghosts


def argparsing():
    parser = argparse.ArgumentParser(description="GhostUMAP Banchmark")
    parser.add_argument(
        "--dataset", "-d", type=str, default="mnist", help="Dataset title"
    )

    args = parser.parse_args()
    return args


def measure_f1(true: np.ndarray, pred: np.ndarray, k=None):
    if k is None:
        k = pred.shape[0]
    true = true[:k]
    pred = pred[:k]
    true_set = set(true)
    pred_set = set(pred)
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return 2 * precision * recall / (precision + recall)


def load_dataset(title):
    data = np.load(f"./data/{title}/{title}.npy")
    knn_indices = np.load(f"./data/{title}/knn_indices.npy")
    knn_dists = np.load(f"./data/{title}/knn_dists.npy")

    return data, (knn_indices, knn_dists)


def benchmark_accuracy(
    data,
    precomputed_knn=None,
    halving_points=[50, 100, 150],
    title="mnist",
    n_ghosts=16,
):
    print("Benchmarking accuracy")

    path = f"./result/{title}_g_{n_ghosts}"
    if not os.path.exists(path):
        os.makedirs(path)

    measures = {
        "f1_elbow": [],
    }
    measures.update({f"f1_{j}": [] for j in range(6)})
    measures.update(
        {
            "ratio_elbow": [],
        }
    )
    measures.update({f"ratio_{j}": [] for j in range(6)})

    ghostumap_ = GhostUMAP(precomputed_knn=precomputed_knn)
    for i in range(10):
        print(f"{title} - Iteration {i + 1}")
        O, G, ghost_indices = ghostumap_.fit_transform(
            data, halving_points=halving_points, n_ghosts=n_ghosts, benchmark="accuracy"
        )

        if not os.path.exists(f"{path}/accuracy"):
            os.makedirs(f"{path}/accuracy")

        np.save(f"{path}/accuracy/O_{i}.npy", O)
        np.save(f"{path}/accuracy/G_{i}.npy", G)
        np.save(f"{path}/accuracy/ghost_indices_{i}.npy", ghost_indices)

        rank_halved, score_halved = ghostumap_.detect_instable_ghosts()
        rank, score = detect_instable_ghosts(O, G, None)

        mean = np.mean(score)
        std = np.std(score)

        for j in range(6):
            e = score[score > mean + (j * std)].shape[0]
            measures[f"f1_{j}"].append(measure_f1(rank, rank_halved, k=e))
            measures[f"ratio_{j}"].append(e / score.shape[0])
        e = score[score > mean + (2.58 * std)].shape[0]
        measures["f1_elbow"].append(measure_f1(rank, rank_halved, k=e))
        measures["ratio_elbow"].append(e / score.shape[0])

    for j in range(6):
        mean = np.mean(measures[f"f1_{j}"])
        std = np.std(measures[f"f1_{j}"])
        mean_ratio = np.mean(measures[f"ratio_{j}"])
        std_ratio = np.std(measures[f"ratio_{j}"])
        measures[f"f1_{j}_mean"] = mean
        measures[f"f1_{j}_std"] = std
        measures[f"ratio_{j}_mean"] = mean_ratio
        measures[f"ratio_{j}_std"] = std_ratio
        print(f"F1_{j}: {mean} ± {std}")
        print(f"Ratio_{j}: {mean_ratio} ± {std_ratio}")

    mean = np.mean(measures["f1_elbow"])
    std = np.std(measures["f1_elbow"])
    mean_ratio = np.mean(measures["ratio_elbow"])
    std_ratio = np.std(measures["ratio_elbow"])

    measures["f1_elbow_mean"] = mean
    measures["f1_elbow_std"] = std
    measures["ratio_elbow_mean"] = mean_ratio
    measures["ratio_elbow_std"] = std_ratio

    print(f"F1_elbow: {mean} ± {std}")
    print(f"Ratio_elbow: {mean_ratio} ± {std_ratio}")

    json.dump(
        measures,
        open(f"{path}/acc_results.json", "w"),
        indent=4,
    )


def benchmark_time(
    data,
    precomputed_knn=(None, None),
    halving_points=[50, 100, 150],
    title="mnist",
    n_ghosts=16,
):
    print(n_ghosts)
    print("Benchmarking time")
    path = f"./result/{title}_g_{n_ghosts}"
    if not os.path.exists(f"{path}"):
        os.makedirs(f"{path}")

    times = {
        "w_halving": [],
        "wo_halving": [],
        "w_halving_optim": [],
        "wo_halving_optim": [],
        "umap": [],
    }

    umap_ = UMAP(precomputed_knn=precomputed_knn)
    ghostumap_ = GhostUMAP(precomputed_knn=precomputed_knn)
    for i in range(10):

        start = time.time()
        umap_.fit_transform(data)
        end = time.time()
        times["umap"].append(end - start)

        start = time.time()
        O, G, ghost_indices = ghostumap_.fit_transform(
            data, halving_points=halving_points, n_ghosts=n_ghosts
        )
        end = time.time()
        times["w_halving"].append(end - start)
        times["w_halving_optim"].append(ghostumap_.time_cost)

        ghostumap_ = GhostUMAP()
        start = time.time()
        O, G, ghost_indices = ghostumap_.fit_transform(data, n_ghosts=n_ghosts)
        end = time.time()
        times["wo_halving"].append(end - start)
        times["wo_halving_optim"].append(ghostumap_.time_cost)

        print(f"Time umap: {times['umap'][-1]}")
        print(f"Time w halving: {times['w_halving'][-1]}")
        print(f"Time wo halving: {times['wo_halving'][-1]}")
        print(f"Time w halving optim: {times['w_halving_optim'][-1]}")
        print(f"Time wo halving optim: {times['wo_halving_optim'][-1]}")

    times["umap_mean"] = np.mean(times["umap"])
    times["umap_std"] = np.std(times["umap"])
    times["w_halving_mean"] = np.mean(times["w_halving"])
    times["w_halving_std"] = np.std(times["w_halving"])
    times["wo_halving_mean"] = np.mean(times["wo_halving"])
    times["wo_halving_std"] = np.std(times["wo_halving"])
    times["w_halving_optim_mean"] = np.mean(times["w_halving_optim"])
    times["w_halving_optim_std"] = np.std(times["w_halving_optim"])
    times["wo_halving_optim_mean"] = np.mean(times["wo_halving_optim"])
    times["wo_halving_optim_std"] = np.std(times["wo_halving_optim"])

    print(f"Umap: {times['umap_mean']} ± {times['umap_std']}")
    print(f"with halving: {times['w_halving_mean']} ± {times['w_halving_std']}")
    print(f"without halving: {times['wo_halving_mean']} ± {times['wo_halving_std']}")
    print(
        f"with halving optim: {times['w_halving_optim_mean']} ± {times['w_halving_optim_std']}"
    )
    print(
        f"without halving optim: {times['wo_halving_optim_mean']} ± {times['wo_halving_optim_std']}"
    )

    json.dump(
        times,
        open(f"{path}/time_results.json", "w"),
        indent=4,
    )


def main():
    # args = argparsing()
    # dataset = args.dataset

    titles = [
        "celegans",
        "cifar10",
        "mnist",
        "fashion_mnist",
        "imdb",
        "20ng",
        "ag_news",
    ]

    print(f"Benchmarking")
    for title in titles[6:]:
        print(f"Dataset: {title}")
        data, precomputed_knn = load_dataset(title)
        if data.shape[0] <= 10000:
            halving_points = [200, 300, 400]
        else:
            halving_points = [50, 100, 150]
        print(halving_points)
        benchmark_accuracy(
            data,
            precomputed_knn,
            halving_points=halving_points,
            title=title,
            n_ghosts=8,
        )
        benchmark_time(
            data,
            precomputed_knn,
            halving_points=halving_points,
            title=title,
            n_ghosts=8,
        )
        benchmark_accuracy(
            data,
            precomputed_knn,
            halving_points=halving_points,
            title=title,
            n_ghosts=16,
        )

        benchmark_time(
            data,
            precomputed_knn,
            halving_points=halving_points,
            title=title,
            n_ghosts=16,
        )

    # # benchmark_accuracy(data, precomputed_knn, title=dataset)
    # benchmark_time(data, title=dataset, halving_points=[200, 300, 400])


if __name__ == "__main__":
    main()
