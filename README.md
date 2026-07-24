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

```python
import numpy as np
from hierarchical_ivf import HierarchicalAdaIVFIndex

rng = np.random.RandomState(42)
base = rng.randn(5000, 64).astype("float32")
base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
base_time = np.arange(len(base)).astype("float32")

index = HierarchicalAdaIVFIndex(
    n_fine_clusters=128,
    n_coarse_clusters=16,
    n_probe=4,
)

index.train(base)
index.add(base, ids=np.arange(len(base)), scalars=base_time)

query = base[-1] + 0.01 * rng.randn(64).astype("float32")
query /= np.linalg.norm(query) + 1e-12

# Search only recent records.
distances, ids = index.search(
    query,
    k=10,
    scalar_range=(4500, 4999),
    use_hierarchy=True,
    early_stop_threshold=1000,
)

print(ids)
print(distances)
```

## Dynamic Insertion Example

```python
new_vectors = rng.randn(1000, 64).astype("float32")
new_vectors /= np.linalg.norm(new_vectors, axis=1, keepdims=True) + 1e-12
new_ids = np.arange(5000, 6000)
new_time = np.arange(5000, 6000).astype("float32")

index.add(new_vectors, ids=new_ids, scalars=new_time, auto_recluster=True)

distances, ids = index.search(
    query,
    k=10,
    scalar_range=(5500, 5999),
    use_hierarchy=True,
)
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
