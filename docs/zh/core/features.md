---
title: 特征定义
description: Torch-RecHub 特征类型定义
---

# 特征定义

Torch-RecHub提供了三种核心特征类，用于处理不同类型的特征：

![特征类型到模型输入示意图](/img/diagrams/feature_types_flow.png)

## DenseFeature

处理数值型特征，如年龄、薪资、日点击量等。

```python
from torch_rechub.basic.features import DenseFeature

# 创建数值型特征
dense_feature = DenseFeature(name="age", embed_dim=1)
```

**参数说明：**
- `name`：特征名称
- `embed_dim`：嵌入向量长度，固定为1

## SparseFeature

处理类别型特征，如城市、学历、性别等。

```python
from torch_rechub.basic.features import SparseFeature

# 创建类别型特征
sparse_feature = SparseFeature(
    name="city",
    vocab_size=100,  # 词汇表大小
    embed_dim=16,     # 嵌入向量长度
    shared_with=None  # 与其他特征共享嵌入表
)
```

**参数说明：**
- `name`：特征名称
- `vocab_size`：词汇表大小
- `embed_dim`：嵌入向量长度，若为None则自动计算
- `shared_with`：共享嵌入表的其他特征名称
- `padding_idx`：填充索引，在InputMask层中会被掩码为0
- `initializer`：嵌入层权重初始化器

## SequenceFeature

处理序列特征或多热特征，如用户行为序列、物品标签等。

```python
from torch_rechub.basic.features import SequenceFeature

# 创建序列特征
sequence_feature = SequenceFeature(
    name="user_history",
    vocab_size=10000,  # 词汇表大小
    embed_dim=32,       # 嵌入向量长度
    pooling="mean"       # 池化方式：mean, sum, concat
)
```

**参数说明：**
- `name`：特征名称
- `vocab_size`：词汇表大小
- `embed_dim`：嵌入向量长度，若为None则自动计算
- `pooling`：池化方式，支持mean、sum、concat
- `shared_with`：共享嵌入表的其他特征名称
- `padding_idx`：填充索引，在InputMask层中会被掩码为0
- `initializer`：嵌入层权重初始化器

## Feature 实例与 Embedding 所有权

> **注意**：`SparseFeature` 和 `SequenceFeature` 会缓存通过 `get_embedding_layer()` 创建的 `nn.Embedding`。如果将**同一个 Feature 实例**传给多个模型，这些模型会使用同一份 Embedding 参数；训练或加载其中一个模型的权重时，其他模型看到的 Embedding 也会随之变化。

下面的两个 EmbeddingLayer 会意外共享 `city` 的 Embedding：

```python
from torch_rechub.basic.features import SparseFeature
from torch_rechub.basic.layers import EmbeddingLayer

features = [SparseFeature(name="city", vocab_size=100, embed_dim=16)]

embedding_a = EmbeddingLayer(features)
embedding_b = EmbeddingLayer(features)

assert embedding_a.embed_dict["city"] is embedding_b.embed_dict["city"]
```

进行独立的模型对比、交叉验证或集成训练时，请为每个模型重新创建 Feature 实例：

```python
def build_features():
    return [SparseFeature(name="city", vocab_size=100, embed_dim=16)]

embedding_a = EmbeddingLayer(build_features())
embedding_b = EmbeddingLayer(build_features())

assert embedding_a.embed_dict["city"] is not embedding_b.embed_dict["city"]
```

仅复制列表并不能解决该问题：`features.copy()` 仍然包含原来的 Feature 实例。这里所说的跨模型意外共享，也不同于在同一个模型内使用 `shared_with` 显式共享 Embedding；后者是预期行为。

## 特征使用示例

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature

# 定义特征
dense_features = [
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1)
]

sparse_features = [
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    SparseFeature(name="gender", vocab_size=3, embed_dim=8)
]

sequence_features = [
    SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling="mean")
]

# 合并所有特征
all_features = dense_features + sparse_features + sequence_features
```
