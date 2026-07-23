---
title: Vector Retrieval Wrapper
description: Torch-RecHub vector retrieval tools
---

# Vector Retrieval Wrapper

Torch-RecHub provides a unified vector retrieval interface that supports three mainstream Approximate Nearest Neighbor (ANN) search libraries: **Annoy**, **FAISS**, and **Milvus**. The standardized Builder-Indexer pattern makes it easy to switch between retrieval backends.

![Vector index Builder-Indexer component diagram](/img/diagrams/vector_index_builder_indexer.png)

## Installation

The current `torch_rechub.serving` package loads all three backends when imported. Even if you use only one backend, you need to install all retrieval extras:

```bash
pip install "torch-rechub[annoy,faiss,milvus]"
```

This is a limitation of the current package import behavior; it does not mean that all three backends must be used at runtime.

## Quick Start

```python
import torch
from torch_rechub.serving import builder_factory

# Prepare embedding vectors
item_embeddings = torch.randn(1000, 64, dtype=torch.float32)  # 1,000 items, 64 dimensions
user_embeddings = torch.randn(10, 64, dtype=torch.float32)    # 10 users, 64 dimensions

# Create a Builder and build the index
builder = builder_factory("faiss", index_type="Flat", metric="L2")

with builder.from_embeddings(item_embeddings) as indexer:
    # Query Top-K
    ids, scores = indexer.query(user_embeddings, top_k=10)
    # Save the index
    indexer.save("item.index")

# Load an existing index
with builder.from_index_file("item.index") as indexer:
    ids, scores = indexer.query(user_embeddings, top_k=10)
```

The meaning of the second return value depends on the metric: for L2/angular it is generally a distance where smaller is closer, while for IP/COSINE it is generally a score where larger is more similar. Do not compare this value directly across different metrics.

## Core Concepts

### Builder-Indexer Pattern

- **Builder**: Manages index construction configuration and is created through the `builder_factory` factory function
- **Indexer**: Performs query and save operations and is obtained through the Builder's context manager

### Factory Function

```python
from torch_rechub.serving import builder_factory

builder = builder_factory(model, **builder_config)
```

| Parameter          | Type   | Description                                                 |
| ------------------ | ------ | ----------------------------------------------------------- |
| `model`            | `str`  | Retrieval backend name: `"annoy"`, `"faiss"`, or `"milvus"` |
| `**builder_config` | `dict` | Configuration arguments passed to the specific Builder      |

---

## Annoy

[Annoy](https://github.com/spotify/annoy) (Approximate Nearest Neighbors Oh Yeah) is an open-source approximate nearest neighbor search library from Spotify. It is memory efficient and supports memory-mapped index files.

### Backend Dependencies

```bash
pip install "torch-rechub[annoy,faiss,milvus]"
```

> ⚠️ **Note**: Annoy includes a C++ extension. If PyPI has no wheel for your Python version and platform, installation falls back to local compilation: Linux/macOS requires a working `gcc`/`g++` or `clang`, while Windows requires [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/). Do not install binary wheels from unknown third-party sources.

### Parameters

```python
builder = builder_factory(
    "annoy",
    d=64,                    # Vector dimension (required)
    metric="angular",        # Distance metric
    n_trees=10,              # Number of trees
    threads=-1,              # Number of build threads
    searchk=-1,              # Number of nodes inspected during search
)
```

| Parameter | Type  | Default     | Description                                                                 |
| --------- | ----- | ----------- | --------------------------------------------------------------------------- |
| `d`       | `int` | Required    | Vector dimension                                                            |
| `metric`  | `str` | `"angular"` | Distance metric: `"angular"` (cosine), `"euclidean"` (Euclidean), or `"dot"` (dot product) |
| `n_trees` | `int` | `10`        | Number of trees to build; more trees improve accuracy but slow construction |
| `threads` | `int` | `-1`        | Number of build threads; `-1` uses all available cores                      |
| `searchk` | `int` | `-1`        | Nodes inspected during search; `-1` means `n_trees * top_k`                 |

### Usage Example

```python
import torch
from torch_rechub.serving import builder_factory

item_embeddings = torch.randn(1000, 64, dtype=torch.float32)
user_embeddings = torch.randn(10, 64, dtype=torch.float32)

# Use cosine similarity
builder = builder_factory(
    "annoy",
    d=64,
    metric="angular",
    n_trees=50,
    searchk=100,
)

with builder.from_embeddings(item_embeddings) as indexer:
    ids, distances = indexer.query(user_embeddings, top_k=10)
    indexer.save("annoy.index")
    
print(f"Retrieved item IDs: {ids}")
print(f"Distances: {distances}")
```

---

## FAISS

[FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) is Meta's open-source high-performance similarity search library with support for multiple index types. The current Torch-RecHub wrapper creates CPU indexes and does not perform CPU-to-GPU index conversion.

### Backend Dependencies

```bash
pip install "torch-rechub[annoy,faiss,milvus]"
```

The `faiss-gpu` package itself provides GPU capabilities, but the current `FaissBuilder` does not move indexes to the GPU. This page's API therefore still uses the CPU path.

### Supported Index Types

| Index Type | Description                       | Use Case                    |
| ---------- | --------------------------------- | --------------------------- |
| `Flat`     | Brute-force search, exact results | Small-scale data (< 100K)   |
| `HNSW`     | Graph-based approximate search    | Medium-scale, high recall   |
| `IVF`      | Inverted index, clustered search  | Large-scale data            |

### Parameters

#### Flat Index

```python
builder = builder_factory(
    "faiss",
    index_type="Flat",       # Index type
    metric="L2",             # Distance metric
)
```

| Parameter    | Type  | Default  | Description                                                 |
| ------------ | ----- | -------- | ----------------------------------------------------------- |
| `index_type` | `str` | `"Flat"` | Index type                                                  |
| `metric`     | `str` | `"L2"`   | Distance metric: `"L2"` (Euclidean) or `"IP"` (inner product) |

#### HNSW Index

```python
builder = builder_factory(
    "faiss",
    index_type="HNSW",
    metric="L2",
    m=32,                    # Maximum number of neighbors per node
    efSearch=50,             # Number of candidate nodes during search
)
```

| Parameter  | Type  | Default | Description                                                      |
| ---------- | ----- | ------- | ---------------------------------------------------------------- |
| `m`        | `int` | `32`    | Maximum neighbors per node; larger values improve accuracy       |
| `efSearch` | `int` | `None`  | Candidate nodes during search; larger values are more accurate but slower |

#### IVF Index

```python
builder = builder_factory(
    "faiss",
    index_type="IVF",
    metric="L2",
    nlists=100,              # Number of cluster centers
    nprobe=10,               # Number of clusters visited during search
)
```

| Parameter | Type  | Default | Description                                                        |
| --------- | ----- | ------- | ------------------------------------------------------------------ |
| `nlists`  | `int` | `100`   | Number of cluster centers; `sqrt(n)` to `4*sqrt(n)` is recommended |
| `nprobe`  | `int` | `None`  | Clusters visited during search; larger values are more accurate but slower |

### Usage Example

```python
import torch
from torch_rechub.serving import builder_factory

item_embeddings = torch.randn(10000, 128, dtype=torch.float32)
user_embeddings = torch.randn(100, 128, dtype=torch.float32)

# Use an HNSW index
builder = builder_factory(
    "faiss",
    index_type="HNSW",
    metric="IP",  # Inner product, suitable for normalized vectors
    m=32,
    efSearch=64,
)

with builder.from_embeddings(item_embeddings) as indexer:
    ids, distances = indexer.query(user_embeddings, top_k=20)
    indexer.save("faiss_hnsw.index")
```

---

## Milvus

[Milvus](https://milvus.io/) is a cloud-native vector database that supports distributed deployment and multiple indexing algorithms, making it suitable for large-scale vector retrieval in production environments.

### Backend Dependencies and Service

```bash
pip install "torch-rechub[annoy,faiss,milvus]"
```

> **Note**: You must also start a Milvus service before using this backend. Milvus Standalone includes additional components and cannot be started by simply running an image without its startup command. Follow the [official Milvus Docker installation guide](https://milvus.io/docs/install_standalone-docker.md) to start and check the service.

### Supported Index Types

| Index Type | Description        | Use Case                   |
| ---------- | ------------------ | -------------------------- |
| `FLAT`     | Brute-force search | Small-scale, exact results |
| `HNSW`     | Graph-based index  | Medium-scale, high recall  |
| `IVF_FLAT` | Inverted index     | Large-scale data           |

### Parameters

#### FLAT Index

```python
builder = builder_factory(
    "milvus",
    d=64,                    # Vector dimension (required)
    index_type="FLAT",
    metric="COSINE",         # Default distance metric
)
```

| Parameter | Type  | Default    | Description                                 |
| --------- | ----- | ---------- | ------------------------------------------- |
| `d`       | `int` | Required   | Vector dimension                            |
| `metric`  | `str` | `"COSINE"` | Distance metric: `"L2"`, `"IP"`, or `"COSINE"` |

#### HNSW Index

```python
builder = builder_factory(
    "milvus",
    d=64,
    index_type="HNSW",
    metric="COSINE",
    m=30,                    # Maximum number of neighbors per node
    ef=50,                   # Number of candidate nodes during search
)
```

| Parameter | Type  | Default | Description                             |
| --------- | ----- | ------- | --------------------------------------- |
| `m`       | `int` | `30`    | Maximum number of neighbors per node    |
| `ef`      | `int` | `None`  | Number of candidate nodes during search |

#### IVF_FLAT Index

```python
builder = builder_factory(
    "milvus",
    d=64,
    index_type="IVF_FLAT",
    metric="IP",
    nlist=128,               # Number of cluster centers
    nprobe=16,               # Number of clusters visited during search
)
```

| Parameter | Type  | Default | Description                              |
| --------- | ----- | ------- | ---------------------------------------- |
| `nlist`   | `int` | `128`   | Number of cluster centers                |
| `nprobe`  | `int` | `None`  | Number of clusters visited during search |

### Usage Example

```python
import torch
from torch_rechub.serving import builder_factory

item_embeddings = torch.randn(10000, 64, dtype=torch.float32)
user_embeddings = torch.randn(100, 64, dtype=torch.float32)

# Use a Milvus HNSW index
builder = builder_factory(
    "milvus",
    d=64,
    index_type="HNSW",
    metric="COSINE",
    m=32,
    ef=64,
)

with builder.from_embeddings(item_embeddings) as indexer:
    ids, distances = indexer.query(user_embeddings, top_k=10)
    # The current wrapper does not support local save
```

`MilvusBuilder.from_embeddings()` creates a randomly named temporary collection and calls `drop()` when the `with` block exits; `save()` and `from_index_file()` are also unimplemented. This wrapper is therefore suitable for one-off experiments, not for managing persistent online collections. For long-running services, create and maintain collections directly with the Milvus client.

---

## Complete Example: Retrieval Model Evaluation

The following is a vector retrieval evaluation example for a single-interest two-tower model, where both user and item embeddings are two-dimensional tensors. Multi-interest models such as MIND and ComiRec, which output `[batch, interests, dim]`, require the interests to be flattened and deduplication and merge strategies to be defined; they cannot use this function directly.

```python
import collections
import numpy as np
import pandas as pd
import torch
from torch_rechub.serving import builder_factory
from torch_rechub.basic.metric import topk_metrics

def match_evaluation(
    user_embedding: torch.Tensor,
    item_embedding: torch.Tensor,
    test_user: dict,
    all_item: dict,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    raw_id_maps: str = "./raw_id_maps.npy",
    topk: int = 10,
    backend: str = "faiss",
    **backend_kwargs,
):
    """
    Evaluate retrieval using vector search
    
    Args:
        user_embedding: User embedding vectors (n_users, dim)
        item_embedding: Item embedding vectors (n_items, dim)
        test_user: Test user data dictionary
        all_item: Complete item data dictionary
        user_col: User ID column name
        item_col: Item ID column name
        raw_id_maps: Path to the ID mapping file
        topk: Number of items to retrieve
        backend: Retrieval backend ("annoy", "faiss", "milvus")
        **backend_kwargs: Additional arguments passed to builder_factory
    
    Returns:
        Evaluation metrics dictionary
    """
    print(f"Evaluating vector retrieval with {backend}")
    
    # 1. Create a Builder
    dim = item_embedding.shape[1]
    
    if backend == "annoy":
        config = {"d": dim, "n_trees": 10}
    elif backend == "faiss":
        config = {"index_type": "Flat", "metric": "L2"}
    elif backend == "milvus":
        config = {"d": dim, "index_type": "FLAT", "metric": "L2"}
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    # Caller arguments override defaults, avoiding duplicate keywords such as index_type
    config.update(backend_kwargs)
    builder = builder_factory(backend, **config)
    
    # 2. Ensure tensors are on the CPU
    item_embedding = item_embedding.cpu().float()
    user_embedding = user_embedding.cpu().float()
    
    # 3. Load ID mappings
    user_map, item_map = np.load(raw_id_maps, allow_pickle=True)
    
    # 4. Build the index and query it
    match_res = collections.defaultdict(dict)
    
    with builder.from_embeddings(item_embedding) as indexer:
        ids, distances = indexer.query(user_embedding, topk)
        ids_np = ids.numpy()
        
        for i, user_id in enumerate(test_user[user_col]):
            items_idx = ids_np[i]
            predicted_item_ids = all_item[item_col][items_idx]
            match_res[user_map[user_id]] = np.vectorize(item_map.get)(predicted_item_ids)
    
    # 5. Build the ground truth
    data = pd.DataFrame({user_col: test_user[user_col], item_col: test_user[item_col]})
    data[user_col] = data[user_col].map(user_map)
    data[item_col] = data[item_col].map(item_map)
    user_pos_item = data.groupby(user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[user_col], user_pos_item[item_col]))
    
    # 6. Compute metrics
    out = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[topk])
    return out


# Usage example
# result = match_evaluation(
#     user_embedding, item_embedding, test_user, all_item,
#     topk=10, backend="faiss", index_type="HNSW", m=32
# )
```

---

## Performance Comparison and Selection Guide

| Feature | Annoy | FAISS | Milvus |
| ------- | ----- | ----- | ------ |
| **Installation Difficulty** | Easy | Moderate | Requires a service |
| **Memory Usage** | Low | Moderate | Depends on service configuration |
| **Build Speed** | Slow | Fast | Fast |
| **Query Speed** | Moderate | Fast | Fast |
| **Current Wrapper GPU Path** | ❌ | ❌ | Determined by the Milvus service |
| **Distributed** | ❌ | ❌ | ✅ |
| **Current Wrapper Use Case** | Small-scale offline | Medium-to-large-scale offline | Temporary experiments |

### Selection Recommendations

- **Quick prototyping / Small datasets**: Use **Annoy** for simple installation and efficient memory use
- **Medium-to-large-scale offline computation**: Use **FAISS**; the current wrapper uses the CPU path
- **Persistent online services**: The Milvus service itself supports distributed operation and dynamic updates, but manage collections directly with the Milvus client instead of using the current context wrapper, which automatically deletes its collection

---

## API Reference

### BaseBuilder

```python
class BaseBuilder(abc.ABC):
    def from_embeddings(self, embeddings: torch.Tensor) -> ContextManager[BaseIndexer]:
        """Build an index from embedding vectors"""
        
    def from_index_file(self, index_file: FilePath) -> ContextManager[BaseIndexer]:
        """Load an index from a file"""
```

### BaseIndexer

```python
class BaseIndexer(abc.ABC):
    def query(self, embeddings: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Query nearest neighbors
        
        Args:
            embeddings: Query vectors (n, d)
            top_k: Number of nearest neighbors to return
            
        Returns:
            ids: Nearest-neighbor IDs (n, top_k)
            distances: Distances (n, top_k)
        """
        
    def save(self, file_path: FilePath) -> None:
        """Save the index to a file"""
```

These are abstract interfaces; not every backend implements the persistence methods. The current Milvus backend raises `NotImplementedError` from `from_index_file()` and `save()`.
