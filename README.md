# HIG: An Update-Friendly Hierarchical IVF Index for Dynamic Range-Filtered ANN Search

HIG is a dynamic hybrid-search index for vector--scalar data. It targets workloads where new vectors are inserted continuously and queries must return approximate nearest neighbors under a scalar range predicate, such as retrieving recent multimedia events by visual similarity and upload time.

The implementation combines:

- **Hierarchical IVF routing**: fine clusters store vectors, while a coarse layer narrows the search scope.
- **Centroid-level graph guidance**: a compact graph over partition representatives repairs coarse-routing misses without maintaining a graph over every vector.
- **Range-aware list layout**: scalar values are stored with vectors so queries can pre-filter candidates by range before distance computation.
- **Append-friendly updates**: new records can be inserted in batches without rebuilding the full index.
## Repository Layout

```text
HIG/
├── include/                 # C++ headers
├── src/                     # C++ extension sources and Python package
│   └── hierarchical_ivf/    # Python API
├── utils/                   # Dataset loading and runtime helpers
├── examples/                # Minimal runnable examples
├── supplementary/           # supplementary material of HIG
├── CMakeLists.txt           # C++ extension build for hierarchical IVF
├── build.sh                 # Convenience build script
├── setup.py                 # Python extension build for Ada-IVF core
└── README.md
```

## Installation

Create or activate a Python environment, then install the package from the repository root:

```bash
cd /path/to/HIG
pip install -e .
```

For best performance, install FAISS if available:

```bash
pip install faiss-cpu
```

To build the C++ extensions in place:

```bash
bash build.sh
```

If the target machine does not support AVX/FMA, disable those flags:

```bash
ENABLE_AVX=OFF bash build.sh
```

## Data Format

HIG expects vector data and scalar attributes:

- `vectors`: a NumPy array with shape `(n, d)` and dtype `float32` or `float64`.
- `scalars`: a one-dimensional NumPy array with shape `(n,)`, storing the scalar attribute used by range filters, such as timestamp, price, or numeric category.
- `ids`: optional record identifiers. If omitted, HIG assigns consecutive integer IDs.

Example:

```python
vectors = np.random.randn(10000, 128).astype("float32")
scalars = np.arange(10000).astype("float32")  # e.g., timestamps
ids = np.arange(10000)
```

For HDF5 datasets, place your loader logic in user code or reuse `utils/h5_loader.py` if the file matches its expected schema.

## Basic Usage

The following example builds an index over synthetic vector--timestamp pairs and runs a range-filtered ANN query. If you run it directly from a source checkout without installing the package, set `PYTHONPATH=src` first.

```python
import numpy as np
from hierarchical_ivf import HierarchicalAdaIVFIndex


def normalize(vectors):
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)


rng = np.random.RandomState(7)
n_base = 800
dim = 32

base = normalize(rng.randn(n_base, dim).astype("float32"))
base_ids = np.arange(n_base)
base_time = np.arange(n_base).astype("float32")

index = HierarchicalAdaIVFIndex(
    n_fine_clusters=32,
    n_coarse_clusters=8,
    n_probe=4,
    max_cluster_size=256,
)

index.train(base)
index.add(base, ids=base_ids, scalars=base_time)

query = base[720].copy()
distances, ids = index.search(
    query,
    k=5,
    scalar_range=(700, 799),
    use_hierarchy=True,
    early_stop_threshold=200,
)

print(ids)
print(distances)
```

## Dynamic Insertion Example

```python
n_insert = 160
new_vectors = normalize(rng.randn(n_insert, dim).astype("float32"))
new_ids = np.arange(n_base, n_base + n_insert)
new_time = np.arange(n_base, n_base + n_insert).astype("float32")

index.add(new_vectors, ids=new_ids, scalars=new_time, auto_recluster=True)

query = new_vectors[10].copy()
distances, ids = index.search(
    query,
    k=5,
    scalar_range=(n_base, n_base + n_insert - 1),
    use_hierarchy=True,
    early_stop_threshold=200,
)

print(ids)
print(distances)
```

## Saving and Loading

```python
index.save("hig_index.pkl")

loaded = HierarchicalAdaIVFIndex()
loaded.load("hig_index.pkl")
```

## Full Example

Run the included demo:

```bash
cd /path/to/HIG
PYTHONIOENCODING=utf-8 PYTHONPATH=src python3 examples/basic_hig_demo.py
```

The demo creates synthetic vector--timestamp data, builds a HIG index, inserts a new batch, and runs a dynamic range-filtered ANN query.

## Notes

- Use smaller `n_fine_clusters` and `n_coarse_clusters` for small datasets; cluster counts must not exceed the number of training vectors.
- `scalar_range=(low, high)` applies an inclusive scalar filter before final top-k ranking.
- Increase `n_probe` or `early_stop_threshold` to improve recall at the cost of more query work.
- Use batch insertion for streaming workloads instead of rebuilding the full index after every update.
