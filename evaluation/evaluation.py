import json
import os
import time

from ghostumap import GhostUMAP
from ghostumap.utils import calculate_variances

from umap import UMAP
import numpy as np
import pandas as pd


def measure_f1(true, pred, k=200):
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


# (n_embeddings, n_ghosts, halving)
candidates = [
    # (1, 16, True),
    # (1, 16, False),
    # (2, 8, True),
    # (2, 8, False),
    # (4, 4, True),
    # (4, 4, False),
    # (8, 2, True),
    # (8, 2, False),
    # (16, 1, False),
    (1, 32, True),
    # (1, 32, False),
    (2, 16, True),
    # (2, 16, False),
    (4, 8, True),
    (4, 8, False),
    (8, 4, True),
    # (8, 4, False),
    (16, 2, True),
    # (16, 2, False),
    (32, 1, False),
]


if __name__ == "__main__":
    print("Loading data...")
    mnist = pd.read_csv("../data/MNIST/mnist.csv")
    data = mnist.drop("label", axis=1).values

    knn_indices = np.load("../data/MNIST/knn_indices.npy")
    knn_dists = np.load("../data/MNIST/knn_dists.npy")
    precomputed_knn = (knn_indices, knn_dists)

    projection_samples = np.load("./result/MNIST/baseline/baseline.npy")
    baseline_rank, _ = calculate_variances(projection_samples)

    ghostumap = GhostUMAP(n_neighbors=15, min_dist=0.1, precomputed_knn=precomputed_knn)
    umap_ = UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, precomputed_knn=precomputed_knn
    )

    columns = [
        "model",
        "n_embeddings",
        "n_ghosts",
        "halving",
        "time_mean",
        "f1_200_mean",
        "f1_500_mean",
        "f1_1000_mean",
        "time_std",
        "f1_200_std",
        "f1_500_std",
        "f1_1000_std",
    ]

    result_df = pd.DataFrame(columns=columns)

    for n_embeddings, n_ghosts, halving in candidates:
        print(f"n_embeddings: {n_embeddings}, n_ghosts: {n_ghosts}, halving: {halving}")
        path = f"./result/MNIST/ghostumap_e_{n_embeddings}_g_{n_ghosts}_h_{'y' if halving else 'n'}"
        if not os.path.exists(path):
            os.makedirs(path)

        if halving:
            halving_ = [40, 60, 80, 100, 120, 140]
        else:
            halving_ = None

        results = {
            "time": [],
            "f1_200": [],
            "f1_500": [],
            "f1_1000": [],
        }

        for i in range(10):
            start = time.time()
            original_embeddings, ghost_embeddings, ghost_indices = (
                ghostumap.fit_transform(
                    data,
                    n_embeddings=n_embeddings,
                    n_ghosts=n_ghosts,
                    halving_points=halving_,
                )
            )
            end = time.time()
            rank, _ = ghostumap.detect_anomalies()

            result = {
                "time": round(end - start, 3),
                "f1_200": measure_f1(baseline_rank, rank, k=200),
                "f1_500": measure_f1(baseline_rank, rank, k=500),
                "f1_1000": measure_f1(baseline_rank, rank, k=1000),
            }

            results["time"].append(result["time"])
            results["f1_200"].append(result["f1_200"])
            results["f1_500"].append(result["f1_500"])
            results["f1_1000"].append(result["f1_1000"])

            np.save(f"{path}/original_embeddings_{i}.npy", original_embeddings)
            np.save(f"{path}/ghost_embeddings_{i}.npy", ghost_embeddings)
            np.save(f"{path}/ghost_indices_{i}.npy", ghost_indices)
            json.dump(result, open(f"{path}/result_{i}.json", "w"))

        row = np.array(
            [
                "GhostUMAP",
                n_embeddings,
                n_ghosts,
                halving,
                round(np.mean(results["time"]), 3),
                round(np.mean(results["f1_200"]), 3),
                round(np.mean(results["f1_500"]), 3),
                round(np.mean(results["f1_1000"]), 3),
                round(np.std(results["time"]), 3),
                round(np.std(results["f1_200"]), 3),
                round(np.std(results["f1_500"]), 3),
                round(np.std(results["f1_1000"]), 3),
            ]
        ).reshape(1, -1)
        df = pd.DataFrame(row.reshape(1, -1), columns=columns)
        df.to_csv(f"{path}/result.csv")
        result_df = pd.concat([result_df, df], ignore_index=True)

    result_df.to_csv("./result/MNIST/result.csv")

    results = {
        "time": [],
        "f1_200": [],
        "f1_500": [],
        "f1_1000": [],
    }

    print("Running baseline...")
    for i in range(10):
        start = time.time()
        baseline = [umap_.fit_transform(data) for _ in range(100)]
        end = time.time()

        baseline = np.array(baseline)
        rank, _ = calculate_variances(baseline)

        result = {
            "time": round(end - start, 3),
            "f1_200": measure_f1(baseline_rank, rank, k=200),
            "f1_500": measure_f1(baseline_rank, rank, k=500),
            "f1_1000": measure_f1(baseline_rank, rank, k=1000),
        }

        results["time"].append(result["time"])
        results["f1_200"].append(result["f1_200"])
        results["f1_500"].append(result["f1_500"])
        results["f1_1000"].append(result["f1_1000"])

        np.save(f"./result/MNIST/baseline/baseline_{i}.npy", baseline)
        json.dump(result, open(f"./result/MNIST/baseline/result_{i}.json", "w"))

    row = np.array(
        [
            "Baseline",
            100,
            0,
            False,
            round(np.mean(results["time"]), 3),
            round(np.mean(results["f1_200"]), 3),
            round(np.mean(results["f1_500"]), 3),
            round(np.mean(results["f1_1000"]), 3),
            round(np.std(results["time"]), 3),
            round(np.std(results["f1_200"]), 3),
            round(np.std(results["f1_500"]), 3),
            round(np.std(results["f1_1000"]), 3),
        ]
    ).reshape(1, -1)

    df = pd.DataFrame(row.reshape(1, -1), columns=columns)
    df.to_csv("./result/MNIST/baseline/result.csv")
    result_df = pd.concat([df, result_df], ignore_index=True)
    result_df.to_csv("./result/MNIST/result.csv")

    print("Finished.")
