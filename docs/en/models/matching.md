---
title: Matching Models
description: Torch-RecHub matching models detailed introduction
---

# Matching Models

Matching models are essential components in recommendation systems, used to quickly retrieve candidate sets relevant to users from massive item catalogs. Torch-RecHub provides various advanced matching models covering different retrieval strategies and modeling approaches.

## 1. DSSM

### Description

DSSM (Deep Structured Semantic Models) is a classic two-tower retrieval model that maps users and items to the same vector space and computes vector similarity for retrieval.

### Paper Reference

```
Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
```

### Core Principles

- **Two-tower Structure**: Contains separate neural networks for user tower and item tower
- **Feature Embedding**: Maps user and item features to low-dimensional vector space
- **Similarity Computation**: Uses cosine similarity or dot product to compute user-item similarity
- **Negative Sampling**: Trains model through negative sampling to optimize ranking performance

### Usage

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.basic.features import SparseFeature, DenseFeature

user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # retained for API compatibility; current DSSM.forward does not use it
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| item_features | list | Item feature list | None |
| temperature | float | Retained parameter; current `DSSM.forward` does not apply it | 1.0 |
| user_params | dict | User tower network parameters | None |
| item_params | dict | Item tower network parameters | None |

### Use Cases

- Text matching scenarios
- Large-scale recommendation systems
- Cold start problems

## 2. FaceBookDSSM

### Description

FaceBookDSSM is a DSSM variant proposed by Facebook, using different network structures and loss functions to further improve retrieval performance.

### Core Principles

- **Two-tower Structure**: Inherits DSSM's two-tower structure
- **Deep Network**: Uses deeper network structure for improved expressiveness
- **Improved Loss Function**: Uses improved loss function for better training
- **Feature Engineering**: Emphasizes feature engineering, supports multiple feature types

### Usage

```python
from torch_rechub.models.matching import FaceBookDSSM

model = FaceBookDSSM(
    user_features=user_features,
    pos_item_features=item_features,
    neg_item_features=neg_item_features,
    user_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
    item_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"}
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| pos_item_features | list | Positive-item feature list | required |
| neg_item_features | list | Negative-item features corresponding to the positive features | required |
| user_params | dict | User tower network parameters | None |
| item_params | dict | Item-tower parameters shared by positive and negative items | None |

`FaceBookDSSM.forward()` returns `(pos_score, neg_score)`. Use it with the pair-wise/BPR loss from `MatchTrainer(mode=1)`, not DSSM's point-wise data format.

### Use Cases

- Large-scale recommendation systems
- Ad retrieval
- Content recommendation

## 3. YoutubeDNN

### Description

YoutubeDNN is a deep retrieval model proposed by YouTube, predicting the next video to watch based on user historical behavior sequences.

### Paper Reference

```
Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
```

### Core Principles

- **Sequence Modeling**: Uses deep neural networks to model user historical behavior sequences
- **Negative Sampling**: Uses negative sampling technique for training efficiency
- **Two-tower Structure**: Contains user and item towers, supports offline item embedding precomputation
- **Multi-objective Optimization**: Simultaneously optimizes multiple objectives like CTR and watch time

### Usage

```python
from torch_rechub.models.matching import YoutubeDNN
from torch_rechub.basic.features import SequenceFeature

user_features = [
    SequenceFeature(name="user_history", vocab_size=100000, embed_dim=32, pooling="mean"),
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=16)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

model = YoutubeDNN(
    user_features=user_features,
    item_features=item_features,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    temperature=0.02
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| item_features | list | Item feature list | None |
| user_params | dict | User tower network parameters | None |
| temperature | float | Temperature parameter | 0.02 |

### Use Cases

- Video recommendation
- Music recommendation
- Content recommendation
- History-based recommendation

## 4. YoutubeSBC

### Description

YoutubeSBC (Sample Bias Correction) is an improved deep retrieval model proposed by YouTube that addresses sampling bias issues.

### Paper Reference

```
Wu, Liang, et al. "RecSys 2019 tutorial: Deep learning for recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

### Core Principles

- **Sample Bias Correction**: Introduces sampling bias correction mechanism for better generalization
- **Two-tower Structure**: Inherits YoutubeDNN's two-tower structure
- **Improved Training**: Uses improved training methods for better performance

### Usage

```python
from torch_rechub.models.matching import YoutubeSBC

model = YoutubeSBC(
    user_features=user_features,
    item_features=item_features,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    temperature=0.02
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| item_features | list | Item feature list | None |
| user_params | dict | User tower network parameters | None |
| temperature | float | Temperature parameter | 0.02 |

### Use Cases

- Large-scale recommendation systems
- Scenarios with severe sampling bias
- Content recommendation

## 5. MIND

### Description

MIND (Multi-Interest Network with Dynamic Routing) is a multi-interest retrieval model that learns multiple interest representations for each user.

### Paper Reference

```
Chen, Jiaxi, et al. "Multi-interest network with dynamic routing for recommendation at Tmall." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

### Core Principles

- **Multi-interest Modeling**: Learns multiple interest vectors for each user
- **Dynamic Routing**: Uses capsule network's dynamic routing mechanism to adaptively aggregate user interests
- **Interest Evolution**: Captures dynamic changes in user interests
- **Two-tower Structure**: Supports offline item embedding precomputation

### Usage

```python
from torch_rechub.models.matching import MIND

user_features = [SparseFeature("user_id", vocab_size=10000, embed_dim=16)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id")
]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16)]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id")
]

model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    interest_num=4,
    temperature=0.02
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | Non-sequential user features | required |
| history_features | list | Item-history sequence with `pooling="concat"` | required |
| item_features | list | Positive item-ID feature | required |
| neg_item_feature | list | Negative item-ID sequence with `pooling="concat"` | required |
| max_length | int | Fixed input-history length | required |
| interest_num | int | Number of interests to learn | 4 |
| temperature | float | Temperature parameter | 0.02 |

### Use Cases

- Diverse user interests scenarios
- E-commerce recommendation
- Content recommendation

## 6. GRU4Rec

### Description

GRU4Rec is a GRU-based sequential recommendation model that captures dynamic dependencies in user behavior sequences.

### Paper Reference

```
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
```

### Core Principles

- **GRU Sequence Modeling**: Uses GRU to capture dynamic changes in user behavior sequences
- **Session Recommendation**: Focuses on within-session recommendation without requiring user history
- **Negative Sampling**: Uses negative sampling for training efficiency
- **BPR Loss**: Uses BPR loss function for ranking optimization

### Usage

```python
from torch_rechub.models.matching import GRU4Rec

user_features = [
    SequenceFeature(name="user_history", vocab_size=100000, embed_dim=32, pooling=None)
]

model = GRU4Rec(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, 16], "dropout": 0.2},
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| history_features | list | History sequences with `pooling="concat"`; only the first is currently used | required |
| item_features | list | Positive item-ID feature | required |
| neg_item_feature | list | Negative item-ID sequence with `pooling="concat"` | required |
| user_params | dict | User MLP; its last dimension must equal the item embedding dimension | required |
| temperature | float | Temperature parameter | 0.02 |

> The current `GRU4Rec.forward()` sums over the wrong axis for list-wise candidates. Treat this as the real constructor interface, not as a stable training tutorial, until the implementation is fixed.

### Use Cases

- Session recommendation
- Short sequence recommendation
- E-commerce scenarios

## 7. NARM

### Description

NARM (Neural Attentive Session-based Recommendation) is an attention-based session recommendation model that captures both local and global interests within sessions.

### Paper Reference

```
Li, Jing, et al. "Neural attentive session-based recommendation." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.
```

### Core Principles

- **GRU Sequence Modeling**: Uses GRU to capture sequence dependencies within sessions
- **Attention Mechanism**: Introduces attention to capture local interests within sessions
- **Global Representation**: Learns global session representation, combining local and global interests
- **Session Recommendation**: Focuses on within-session recommendation

### Usage

```python
from torch_rechub.models.matching import NARM

model = NARM(
    item_history_feature=item_history_feature,
    hidden_dim=64,
    emb_dropout_p=0.2,
    session_rep_dropout_p=0.2,
    item_feature=item_feature,  # only needed for in-batch negatives/item-tower inference
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | Session item sequence; token 0 is padding | required |
| hidden_dim | int | GRU hidden dimension | required |
| emb_dropout_p | float | Item-embedding dropout | required |
| session_rep_dropout_p | float | Session-representation dropout | required |
| item_feature | SparseFeature or None | Optional target item ID for an independent item tower/in-batch negatives | None |

### Use Cases

- Session recommendation
- Short sequence recommendation
- Local and global interest modeling

## 8. SASRec

### Description

SASRec (Self-Attentive Sequential Recommendation) is a self-attention based sequential recommendation model that captures long-range dependencies in user behavior sequences.

### Paper Reference

```
Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.
```

### Core Principles

- **Self-attention Mechanism**: Uses multi-head self-attention to capture long-range dependencies
- **Positional Encoding**: Adds positional information to preserve sequence order
- **Layer Normalization**: Accelerates convergence and improves training stability
- **Residual Connection**: Enhances model expressiveness

### Usage

```python
from torch_rechub.models.matching import SASRec

model = SASRec(
    features=[seq_feature, pos_feature, neg_feature],
    max_len=50,
    dropout_rate=0.2,
    num_blocks=2,
    num_heads=4,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | History, positive, then negative sequence features; all are `SequenceFeature` | required |
| max_len | int | Sequence length | 50 |
| dropout_rate | float | Embedding/attention/FFN dropout | 0.5 |
| num_blocks | int | Number of self-attention blocks | 2 |
| num_heads | int | Attention heads; must divide the embedding dimension | 1 |
| item_feature | SparseFeature or None | Experimental independent item-tower feature | None |

> SASRec currently creates its mask/positions on CPU, and the `item_feature` branch accesses a nonexistent embedding attribute. This page therefore documents the real constructor without claiming the GPU or item-tower paths are usable.

### Use Cases

- Long sequence recommendation
- Scenarios where user behavior sequences are important
- Sequential recommendation tasks

## 9. SINE

### Description

SINE (Sparse Interest Network for Sequential Recommendation) is a sparse interest network that effectively models users' sparse interests.

### Paper Reference

```
Chen, Jiaxi, et al. "SINE: A sparse interest network for sequential recommendation." Proceedings of the 14th ACM Conference on Recommender Systems. 2021.
```

### Core Principles

- **Sparse Interest Modeling**: Specifically handles sparse user interest problems
- **Dynamic Routing**: Uses dynamic routing mechanism to adaptively aggregate user interests
- **Interest Evolution**: Captures dynamic changes in user interests
- **Efficient Computation**: Optimized for computational efficiency, suitable for large-scale data

### Usage

```python
from torch_rechub.models.matching import SINE

model = SINE(
    history_features=["hist_item_id"],
    item_features=["item_id"],
    neg_item_features=["neg_items"],
    num_items=100000,
    embedding_dim=32,
    hidden_dim=64,
    num_concept=100,
    num_intention=4,
    seq_max_len=50,
    num_heads=4,
    temperature=0.02
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| history_features | list[str] | History item-sequence field names; only the first is currently used | required |
| item_features | list[str] | Positive-item field names | required |
| neg_item_features | list[str] | Negative-item sequence field names | required |
| num_items | int | Item vocabulary size including padding | required |
| embedding_dim | int | Embedding dimension | required |
| hidden_dim | int | Attention hidden dimension | required |
| num_concept | int | Number of global concept prototypes | required |
| num_intention | int | Number of intentions selected per user | required |
| seq_max_len | int | Fixed input sequence length | required |
| num_heads | int | Self-attention heads | 1 |
| temperature | float | Temperature parameter | 0.02 |

### Use Cases

- Scenarios with sparse user interests
- Large-scale recommendation systems
- E-commerce recommendation

## 10. STAMP

### Description

STAMP (Short-Term Attention/Memory Priority Model) is an attention-based session recommendation model that focuses on modeling recent user behavior.

### Paper Reference

```
Liu, Qiao, et al. "STAMP: short-term attention/memory priority model for session-based recommendation." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

### Core Principles

- **Short-term Attention**: Focuses on recent user behavior, giving higher weight to recent actions
- **Memory Module**: Maintains a memory vector to capture global session information
- **Session Recommendation**: Focuses on within-session recommendation
- **Simple and Efficient**: Simple model structure with high computational efficiency

### Usage

```python
from torch_rechub.models.matching import STAMP

model = STAMP(
    item_history_feature=item_history_feature,
    weight_std=0.05,
    emb_std=0.05,
    item_feature=item_feature,  # only needed for in-batch negatives/item-tower inference
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | Session item sequence; token 0 is padding | required |
| weight_std | float | Initialization standard deviation for attention parameters | required |
| emb_std | float | Initialization standard deviation for embeddings/linear layers | required |
| item_feature | SparseFeature or None | Optional target item ID for item-tower inference/in-batch negatives | None |

### Use Cases

- Session recommendation
- Short-term interest modeling
- E-commerce scenarios

## 11. ComirecDR

### Description

ComirecDR (Controllable Multi-Interest Recommendation with Dynamic Routing) is a controllable multi-interest recommendation model that allows controlling the number of generated interests.

### Paper Reference

```
Chen, Jiaxi, et al. "Controllable multi-interest framework for recommendation." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
```

### Core Principles

- **Controllable Multi-Interest**: Allows controlling the number of generated interests
- **Dynamic Routing**: Uses dynamic routing mechanism to adaptively aggregate user interests
- **Two-tower Structure**: Supports offline item embedding precomputation
- **Efficient Computation**: Optimized for computational efficiency, suitable for large-scale data

### Usage

```python
from torch_rechub.models.matching import ComirecDR

model = ComirecDR(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    interest_num=4,
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | Non-sequential user features | required |
| history_features | list | Item history with `pooling="concat"` | required |
| item_features | list | Positive-item features | required |
| neg_item_feature | list | Negative-item sequence with `pooling="concat"` | required |
| max_length | int | Fixed history length | required |
| interest_num | int | Number of interests | 4 |
| temperature | float | Temperature parameter | 0.02 |

### Use Cases

- Controllable multi-interest recommendation
- Scenarios with diverse user interests
- Large-scale recommendation systems

## 12. ComirecSA

### Description

ComirecSA (Controllable Multi-Interest Recommendation with Self-Attention) is the self-attention version of Comirec, using self-attention mechanism to model user interests.

### Core Principles

- **Self-attention Mechanism**: Uses self-attention to capture dependencies in user behavior sequences
- **Controllable Multi-Interest**: Allows controlling the number of generated interests
- **Two-tower Structure**: Supports offline item embedding precomputation
- **Efficient Computation**: Optimized for computational efficiency, suitable for large-scale data

### Usage

```python
from torch_rechub.models.matching import ComirecSA

model = ComirecSA(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    interest_num=4,
    temperature=0.02,
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | Non-sequential user features | required |
| history_features | list | Item history with `pooling="concat"` | required |
| item_features | list | Positive-item features | required |
| neg_item_feature | list | Negative-item sequence with `pooling="concat"` | required |
| interest_num | int | Number of interests | 4 |
| temperature | float | Temperature parameter | 0.02 |

### Use Cases

- Controllable multi-interest recommendation
- Long sequence interest modeling
- Large-scale recommendation systems

## 13. Model Comparison

| Model | Complexity | Expressiveness | Efficiency | Use Cases |
| --- | --- | --- | --- | --- |
| DSSM | Low | Medium | High | Text matching, cold start |
| FaceBookDSSM | Medium | High | Medium | Large-scale recommendation, ad retrieval |
| YoutubeDNN | Medium | High | Medium | Video recommendation, content recommendation |
| YoutubeSBC | Medium | High | Medium | Large-scale recommendation, sampling bias scenarios |
| MIND | Medium | High | Medium | Multi-interest recommendation, e-commerce |
| GRU4Rec | Medium | Medium | High | Session recommendation, short sequence |
| NARM | Medium | Medium | Medium | Session recommendation, local/global interests |
| SASRec | High | High | Low | Long sequence recommendation, sequential recommendation |
| SINE | High | High | Medium | Sparse interest modeling, large-scale recommendation |
| STAMP | Low | Medium | High | Session recommendation, short-term interests |
| ComirecDR | Medium | High | Medium | Controllable multi-interest, large-scale recommendation |
| ComirecSA | High | High | Medium | Controllable multi-interest, long sequence recommendation |

## 14. Usage Recommendations

1. **Choose models based on data characteristics**:
   - For long sequence data, use SASRec, ComirecSA
   - For session data, use GRU4Rec, NARM, STAMP
   - For multi-interest scenarios, use MIND, ComirecDR, ComirecSA

2. **Choose models based on computational resources**:
   - With limited resources, use DSSM, GRU4Rec, STAMP
   - With sufficient resources, try more complex models like SASRec, SINE

3. **Consider business requirements**:
   - For controllable multi-interest, use ComirecDR, ComirecSA
   - For sparse interest handling, use SINE
   - For sampling bias correction, use YoutubeSBC

4. **Try combining multiple retrieval strategies**:
   - Different models may capture different user interests; model fusion can improve overall retrieval performance
   - Combine content-based retrieval, collaborative filtering, and other strategies

## 15. Complete Training Example

```python
import os

from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 1. Define features
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1),
    SparseFeature(name="gender", vocab_size=3, embed_dim=8)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# 2. Prepare data
# Assume x and y are preprocessed feature and label data
# x contains user features and item features
x = {
    "user_id": user_id_data,
    "age": age_data,
    "gender": gender_data,
    "item_id": item_id_data,
    "category": category_data
}
y = label_data  # click/no-click labels

# Test user data and all item data
x_test_user = {
    "user_id": test_user_id_data,
    "age": test_age_data,
    "gender": test_gender_data
}
x_all_item = {
    "item_id": all_item_id_data,
    "category": all_item_category_data
}

# 3. Create data generator
dg = MatchDataGenerator(x, y)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    x_test_user=x_test_user,
    x_all_item=x_all_item,
    batch_size=256,
    num_workers=8
)

# 4. Create model
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # current DSSM.forward does not use this parameter
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"}
)

# 5. Create trainer
trainer = MatchTrainer(
    model=model,
    mode=0,  # 0: point-wise, 1: pair-wise, 2: list-wise
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cpu",
    model_path="saved/dssm"
)

# 6. Train model
os.makedirs("saved/dssm", exist_ok=True)
# test_dl/item_dl are unlabeled embedding-inference loaders, not validation loaders
trainer.fit(train_dl)

# 7. Export ONNX model (first install: pip install "torch-rechub[onnx]")
# Export user tower
trainer.export_onnx("user_tower.onnx", mode="user")
# Export item tower
trainer.export_onnx("item_tower.onnx", mode="item")

# 8. Vector retrieval example
# Generate user embeddings
user_embeddings = trainer.inference_embedding(
    model, mode="user", data_loader=test_dl, model_path="saved/dssm"
)
# Generate item embeddings
item_embeddings = trainer.inference_embedding(
    model, mode="item", data_loader=item_dl, model_path="saved/dssm"
)

# Use Annoy or Faiss for vector indexing and retrieval
# Here's an example using Annoy
# First install: pip install "torch-rechub[annoy]"
from annoy import AnnoyIndex

# Create index
index = AnnoyIndex(64, 'angular')  # 64 is the embedding dimension
for i, embedding in enumerate(item_embeddings):
    index.add_item(i, embedding.tolist())
index.build(10)  # 10 trees

# Retrieval example
user_idx = 0
user_emb = user_embeddings[user_idx].tolist()
recall_results = index.get_nns_by_vector(user_emb, 10)  # Retrieve top 10 items
print(f"User {user_idx} retrieved items: {recall_results}")
```

## 16. FAQ

### Q: How to handle large-scale item sets?
A: Try the following approaches:
- Use two-tower structure to support offline item embedding precomputation
- Use approximate nearest neighbor search libraries (e.g., Annoy, Faiss) to accelerate vector retrieval
- Adopt hierarchical retrieval strategy: coarse retrieval first, then fine retrieval

### Q: How to handle cold start problems?
A: Try the following approaches:
- For new users, use content-based retrieval
- For new items, use collaborative filtering or content-based retrieval
- Use transfer learning to transfer knowledge from related domains

### Q: How to evaluate matching model performance?
A: Common retrieval evaluation metrics include:
- Recall@K: Proportion of relevant items in top K results
- Precision@K: Proportion of relevant items in top K results
- NDCG@K: Retrieval quality considering ranking
- Hit@K: Whether at least one relevant item is retrieved
- MRR@K: Mean Reciprocal Rank

### Q: How to choose the right negative sampling strategy?
A: Common negative sampling strategies include:
- Random negative sampling: Simple and efficient, but may sample irrelevant items
- Popularity-based negative sampling: Samples based on item popularity, more realistic
- Hard negative sampling: Samples negatives similar to positives, improves model discrimination
- Contrastive learning negative sampling: Methods like MoCo, SimCLR

### Q: How to optimize matching model performance?
A: Try the following approaches:
- Increase model depth and width to improve expressiveness
- Use more advanced feature engineering
- Optimize negative sampling strategy
- Try model fusion
- Adjust temperature parameter to optimize similarity distribution

## 17. Deployment Suggestions

1. **Offline Precomputation**: For two-tower models, precompute item embeddings offline to reduce online computation
2. **Vector Indexing**: Use efficient vector indexing libraries (e.g., Faiss) to store item embeddings and accelerate online retrieval
3. **Hierarchical Deployment**: Adopt hierarchical retrieval architecture - coarse retrieval with simple models first, then fine retrieval with complex models
4. **Caching Mechanism**: Cache retrieval results for high-frequency users to reduce redundant computation
5. **Real-time Updates**: Periodically update item embeddings to ensure retrieval result freshness
6. **A/B Testing**: Validate different retrieval strategies through A/B testing

Matching models are essential components of recommendation systems. Choosing the right matching model and optimizing it can significantly improve overall recommendation performance. Torch-RecHub provides a rich set of matching models covering different retrieval strategies, making it easy for developers to select and use based on business requirements.
