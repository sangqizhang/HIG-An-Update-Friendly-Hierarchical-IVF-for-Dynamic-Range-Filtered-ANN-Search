#!/usr/bin/env python3
"""Minimal HIG example: build, insert, and search with a scalar range filter."""

from __future__ import print_function

import os
import sys

sys.dont_write_bytecode = True

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from hierarchical_ivf import HierarchicalAdaIVFIndex


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-12)


def main():
    rng = np.random.RandomState(7)
    dim = 32
    n_base = 800
    n_insert = 160

    base_vectors = normalize(rng.randn(n_base, dim).astype("float32"))
    base_ids = np.arange(n_base)
    base_timestamps = np.arange(n_base).astype("float32")

    index = HierarchicalAdaIVFIndex(
        n_fine_clusters=32,
        n_coarse_clusters=8,
        n_probe=4,
        max_cluster_size=256,
    )

    print("Training HIG...")
    index.train(base_vectors)

    print("Adding base vectors...")
    index.add(base_vectors, ids=base_ids, scalars=base_timestamps)

    new_vectors = normalize(rng.randn(n_insert, dim).astype("float32"))
    new_ids = np.arange(n_base, n_base + n_insert)
    new_timestamps = np.arange(n_base, n_base + n_insert).astype("float32")

    print("Adding streaming batch...")
    index.add(new_vectors, ids=new_ids, scalars=new_timestamps, auto_recluster=True)

    query = new_vectors[10] + 0.01 * rng.randn(dim).astype("float32")
    query = query / (np.linalg.norm(query) + 1e-12)

    recent_range = (n_base, n_base + n_insert - 1)
    print("Searching recent time window: {}".format(recent_range))
    distances, result_ids = index.search(
        query,
        k=5,
        scalar_range=recent_range,
        use_hierarchy=True,
        early_stop_threshold=200,
    )

    print("Result IDs:", result_ids)
    print("Distances:", distances)
    print("Index stats:", index.get_stats())


if __name__ == "__main__":
    main()
