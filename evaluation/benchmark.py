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
    data, precomputed_knn=None, halving_points=[50, 100, 150], title="mnist"
):
    print("Benchmarking accuracy")

    if not os.path.exists(f"./result/{title}"):
        os.makedirs(f"./result/{title}")

    measures = {
        "f1_5": [],
        "f1_4": [],
        "f1_3": [],
        "f1_2": [],
        "f1_1": [],
        "f1_elbow": [],
    }

    ghostumap_ = GhostUMAP(precomputed_knn=precomputed_knn)
    for i in range(10):
        print(f"{title} - Iteration {i + 1}")
        O, G, ghost_indices = ghostumap_.fit_transform(
            data, halving_points=halving_points, n_ghosts=16, benchmark="accuracy"
        )

        if not os.path.exists(f"./result/{title}/accuracy"):
            os.makedirs(f"./result/{title}/accuracy")

        np.save(f"./result/{title}/accuracy/O_{i}.npy", O)
        np.save(f"./result/{title}/accuracy/G_{i}.npy", G)
        np.save(f"./result/{title}/accuracy/ghost_indices_{i}.npy", ghost_indices)

        rank_halved, score_halved = ghostumap_.detect_instable_ghosts()
        rank, score = detect_instable_ghosts(O, G, None)

        for j in range(1, 6):
            measures[f"f1_{j}"].append(
                measure_f1(rank, rank_halved, k=rank.shape[0] * j // 100)
            )
        measures["f1_elbow"].append(
            measure_f1(rank, rank_halved, k=score[score > np.mean(score)].shape[0])
        )

    for j in range(1, 6):
        mean = np.mean(measures[f"f1_{j}"])
        std = np.std(measures[f"f1_{j}"])
        measures[f"f1_{j}_mean"] = mean
        measures[f"f1_{j}_std"] = std
        print(f"F1_@{j}%: {mean} ± {std}")

    f1_elbow_mean = np.mean(measures["f1_elbow"])
    f1_elbow_std = np.std(measures["f1_elbow"])
    measures["f1_elbow_mean"] = f1_elbow_mean
    measures["f1_elbow_std"] = f1_elbow_std
    print(f"F1_elbow: {f1_elbow_mean} ± {f1_elbow_std}")

    json.dump(
        measures,
        open(f"./result/{title}/acc_results.json", "w"),
        indent=4,
    )


def benchmark_time(
    data, precomputed_knn=(None, None), halving_points=[50, 100, 150], title="mnist"
):
    print("Benchmarking time")
    if not os.path.exists(f"./result/{title}"):
        os.makedirs(f"./result/{title}")

    times = {
        "w_halving": [],
        "wo_halving": [],
        # "w_halving_optim": [],
        # "wo_halving_optim": [],
        "umap": [],
    }

    umap_ = UMAP(precomputed_knn=precomputed_knn)
    ghostumap_ = GhostUMAP(precomputed_knn=precomputed_knn)
    for i in range(2):

        start = time.time()
        umap_.fit_transform(data)
        end = time.time()
        times["umap"].append(end - start)

        start = time.time()
        O, G, ghost_indices = ghostumap_.fit_transform(
            data, halving_points=halving_points, n_ghosts=16
        )
        end = time.time()
        # times["w_halving"].append(end - start)
        # times["w_halving_optim"].append(ghostumap_.time_cost)
        times["w_halving"].append(ghostumap_.time_cost)

        ghostumap_ = GhostUMAP()
        start = time.time()
        O, G, ghost_indices = ghostumap_.fit_transform(data, n_ghosts=16)
        end = time.time()
        # times["wo_halving"].append(end - start)
        # times["wo_halving_optim"].append(ghostumap_.time_cost)
        times["wo_halving"].append(ghostumap_.time_cost)

        print(f"Time umap: {times['umap'][-1]}")
        print(f"Time w halving: {times['w_halving'][-1]}")
        print(f"Time wo halving: {times['wo_halving'][-1]}")
        # print(f"Time w halving optim: {times['w_halving_optim'][-1]}")
        # print(f"Time wo halving optim: {times['wo_halving_optim'][-1]}")

    times["umap_mean"] = np.mean(times["umap"])
    times["umap_std"] = np.std(times["umap"])
    times["w_halving_mean"] = np.mean(times["w_halving"])
    times["w_halving_std"] = np.std(times["w_halving"])
    times["wo_halving_mean"] = np.mean(times["wo_halving"])
    times["wo_halving_std"] = np.std(times["wo_halving"])
    # times["w_halving_optim_mean"] = np.mean(times["w_halving_optim"])
    # times["w_halving_optim_std"] = np.std(times["w_halving_optim"])
    # times["wo_halving_optim_mean"] = np.mean(times["wo_halving_optim"])
    # times["wo_halving_optim_std"] = np.std(times["wo_halving_optim"])

    print(f"Umap: {times['umap_mean']} ± {times['umap_std']}")
    print(f"with halving: {times['w_halving_mean']} ± {times['w_halving_std']}")
    print(f"without halving: {times['wo_halving_mean']} ± {times['wo_halving_std']}")
    # print(
    #     f"with halving optim: {times['w_halving_optim_mean']} ± {times['w_halving_optim_std']}"
    # )
    # print(
    #     f"without halving optim: {times['wo_halving_optim_mean']} ± {times['wo_halving_optim_std']}"
    # )

    json.dump(
        times,
        open(f"./result/{title}/time_results.json", "w"),
        indent=4,
    )


def main():
    args = argparsing()
    dataset = args.dataset

    print(f"Loading {dataset} dataset")
    data, precomputed_knn = load_dataset(dataset)
    # benchmark_accuracy(data, precomputed_knn, title=dataset)
    benchmark_time(data, title=dataset, halving_points=[200, 300, 400])


if __name__ == "__main__":
    main()
