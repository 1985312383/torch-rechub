---
title: Online Serving Example
description: Torch-RecHub online serving deployment example
---

# Online Serving Example

This page demonstrates the basic way to connect Torch-RecHub deployment components. The example does not include an API gateway, concurrency control, monitoring, or persistent service orchestration.

## Ranking Model Deployment

### 1. Export the ONNX Model

```python
from torch_rechub.trainers import CTRTrainer

# Export after training
trainer.export_onnx("deepfm.onnx", dynamic_batch=True)
```

### 2. ONNX Runtime Inference

```python
import numpy as np
import onnxruntime as ort

# Load the model
session = ort.InferenceSession("deepfm.onnx")

# Prepare inputs
inputs = {
    "city": np.array([1, 2, 3], dtype=np.int64),
    "age": np.array([0.5, 0.3, 0.8], dtype=np.float32),
}

# Run inference
outputs = session.run(None, inputs)
predictions = outputs[0]
```

## Retrieval Model Deployment

### 1. Export the Two-Tower Model

```python
from torch_rechub.trainers import MatchTrainer

# Export the user and item towers separately
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

### 2. Build the Item Index Offline

```python
import numpy as np
import onnxruntime as ort

# Load the item tower
item_session = ort.InferenceSession("item_tower.onnx")

# Compute embeddings for all items
item_embeddings = []
for batch in item_dataloader:
    inputs = {k: v.numpy() for k, v in batch.items()}
    emb = item_session.run(None, inputs)[0]
    item_embeddings.append(emb)

item_embeddings = np.concatenate(item_embeddings)
```

### 3. Online Retrieval Service (Milvus)

The unified factory currently imports all three backends at once, so first install them with `pip install "torch-rechub[annoy,faiss,milvus]"` and start a reachable Milvus service.

```python
import torch
import numpy as np
import onnxruntime as ort
from torch_rechub.serving import builder_factory

# Load the user tower
user_session = ort.InferenceSession("user_tower.onnx")

# Connect to the Milvus service
embed_dim = 64  # Embedding dimension
builder = builder_factory(
    "milvus",
    d=embed_dim,
    index_type="HNSW",
    metric="IP",
    host="localhost",
    port=19530
)

# Write item embeddings to Milvus
with builder.from_embeddings(torch.tensor(item_embeddings, dtype=torch.float32)) as indexer:
    # Compute the user embedding
    user_inputs = {"user_id": np.array([123], dtype=np.int64)}
    user_emb = user_session.run(None, user_inputs)[0]

    # Perform vector retrieval
    ids, scores = indexer.query(torch.tensor(user_emb, dtype=torch.float32), top_k=100)
    recall_items = ids[0].tolist()
```

> This use of `MilvusBuilder.from_embeddings()` is suitable only for demonstration: it creates a temporary collection and deletes it when the `with` block exits. The current wrapper also cannot manage a persistent Milvus collection through `save()` or `from_index_file()`; production services should manage long-lived collections directly with the Milvus client.

## Best Practices

1. **Model Quantization**: Use INT8/FP16 quantization to reduce inference latency
2. **Batch Inference**: Combine requests into batches to improve throughput
3. **Index Preloading**: Preload the index into memory when the service starts
4. **Monitoring and Alerting**: Monitor inference latency and error rates
