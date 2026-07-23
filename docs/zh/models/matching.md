---
title: 召回模型
description: Torch-RecHub 召回模型详细介绍
---

# 召回模型

召回模型是推荐系统中的重要组成部分，用于从海量物品中快速召回与用户相关的候选集。Torch-RecHub 提供了多种先进的召回模型，涵盖了不同的召回策略和建模方式。

## 1. DSSM

### 功能描述

DSSM（Deep Structured Semantic Models）是一种经典的双塔召回模型，通过将用户和物品映射到同一向量空间，计算向量相似度来进行召回。

### 论文引用

```
Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
```

### 核心原理

- **双塔结构**：包含用户塔和物品塔两个独立的神经网络
- **特征嵌入**：将用户和物品特征分别映射到低维向量空间
- **相似度计算**：使用余弦相似度或点积计算用户和物品向量的相似度
- **负采样**：通过负采样训练模型，优化排序性能

### 使用方法

```python
import os

from torch_rechub.models.matching import DSSM
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 定义特征
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# 创建模型
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # 当前 DSSM forward 尚未使用该参数
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | None |
| item_features | list | 物品特征列表 | None |
| temperature | float | 保留参数；当前 DSSM `forward` 未将它应用到分数 | 1.0 |
| user_params | dict | 用户塔网络参数 | None |
| item_params | dict | 物品塔网络参数 | None |

### 适用场景

- 文本匹配场景
- 大规模推荐系统
- 冷启动问题

## 2. FaceBookDSSM

### 功能描述

FaceBookDSSM 是 Facebook 提出的 DSSM 变种，采用了不同的网络结构和损失函数，进一步提高了召回效果。

### 核心原理

- **双塔结构**：继承了 DSSM 的双塔结构
- **深层网络**：使用更深的网络结构，提高模型的表达能力
- **改进的损失函数**：使用改进的损失函数，优化模型训练
- **特征工程**：注重特征工程，支持多种特征类型

### 使用方法

```python
from torch_rechub.models.matching import FaceBookDSSM

# 创建模型
model = FaceBookDSSM(
    user_features=user_features,
    pos_item_features=item_features,
    neg_item_features=neg_item_features,
    user_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
    item_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | None |
| pos_item_features | list | 正样本物品特征列表 | 必填 |
| neg_item_features | list | 与正样本对应的负样本特征列表 | 必填 |
| user_params | dict | 用户塔网络参数 | None |
| item_params | dict | 正负物品共用的物品塔参数 | None |

`FaceBookDSSM.forward()` 返回 `(pos_score, neg_score)`，应配合 `MatchTrainer(mode=1)` 的 pair-wise/BPR 损失，不是 DSSM 的 point-wise 数据格式。

### 适用场景

- 大规模推荐系统
- 广告召回
- 内容推荐

## 3. YoutubeDNN

### 功能描述

YoutubeDNN 是 YouTube 提出的深度召回模型，基于用户历史行为序列预测下一个观看的视频。

### 论文引用

```
Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
```

### 核心原理

- **序列建模**：使用深度神经网络建模用户历史行为序列
- **负采样**：采用负采样技术，提高训练效率
- **双塔结构**：包含用户塔和物品塔，支持离线预计算物品嵌入
- **多目标优化**：同时优化点击率和观看时长等多个目标

### 使用方法

```python
from torch_rechub.models.matching import YoutubeDNN
from torch_rechub.basic.features import SequenceFeature

# 定义序列特征
user_features = [
    SequenceFeature(name="user_history", vocab_size=100000, embed_dim=32, pooling="mean"),
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=16)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# 创建模型
model = YoutubeDNN(
    user_features=user_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    temperature=0.02
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | None |
| item_features | list | 物品特征列表 | None |
| neg_item_feature | list | 负样本 ID 序列，需使用 `pooling="concat"` 并与正样本 ID 共享 embedding | 必填 |
| user_params | dict | 用户塔网络参数 | None |
| temperature | float | 温度参数 | 0.02 |

### 适用场景

- 视频推荐
- 音乐推荐
- 内容推荐
- 基于历史行为的推荐

## 4. YoutubeSBC

### 功能描述

YoutubeSBC（Sample Bias Correction）是 YouTube 提出的改进版深度召回模型，解决了采样偏差问题。

### 论文引用

```
Wu, Liang, et al. "RecSys 2019 tutorial: Deep learning for recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

### 核心原理

- **采样偏差校正**：引入采样偏差校正机制，提高模型的泛化能力
- **双塔结构**：继承了 YoutubeDNN 的双塔结构
- **改进的训练方法**：使用改进的训练方法，优化模型性能

### 使用方法

```python
from torch_rechub.models.matching import YoutubeSBC

# 创建模型
model = YoutubeSBC(
    user_features=user_features,
    item_features=item_features,
    sample_weight_feature=sample_weight_feature,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    batch_size=256,
    n_neg=3,
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | None |
| item_features | list | 物品特征列表 | None |
| sample_weight_feature | list | 用于采样偏差校正的物品采样权重特征 | 必填 |
| user_params | dict | 用户塔网络参数 | None |
| item_params | dict | 物品塔网络参数 | None |
| batch_size | int | 必须与 DataLoader 的 batch size 一致 | 必填 |
| n_neg | int | 每个正样本的 batch 内负样本数，必须小于 batch size | 3 |
| temperature | float | 温度参数 | 0.02 |

### 适用场景

- 大规模推荐系统
- 采样偏差问题严重的场景
- 内容推荐

## 5. MIND

### 功能描述

MIND（Multi-Interest Network with Dynamic Routing）是一种多兴趣召回模型，能够为每个用户学习多个兴趣表示。

### 论文引用

```
Chen, Jiaxi, et al. "Multi-interest network with dynamic routing for recommendation at Tmall." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

### 核心原理

- **多兴趣建模**：为每个用户学习多个兴趣向量
- **动态路由**：使用胶囊网络中的动态路由机制，自适应地聚合用户兴趣
- **兴趣演化**：捕捉用户兴趣的动态变化
- **双塔结构**：支持离线预计算物品嵌入

### 使用方法

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

# 创建模型
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

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 非序列用户特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史物品序列 | 必填 |
| item_features | list | 正样本物品 ID 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 ID 序列 | 必填 |
| max_length | int | 输入历史序列的固定长度 | 必填 |
| interest_num | int | 学习的兴趣数量 | 4 |
| temperature | float | 温度参数 | 0.02 |

### 适用场景

- 用户兴趣多样化的场景
- 电商推荐
- 内容推荐

## 6. GRU4Rec

### 功能描述

GRU4Rec 是一种基于 GRU（Gated Recurrent Unit）的序列推荐模型，能够捕获用户行为序列中的动态依赖关系。

### 论文引用

```
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
```

### 核心原理

- **GRU 序列建模**：使用 GRU 捕捉用户行为序列的动态变化
- **会话推荐**：专注于会话内的推荐，不需要用户历史数据
- **负采样**：采用负采样技术，提高训练效率
- **BPR 损失**：使用 BPR 损失函数，优化排序性能

### 使用方法

```python
from torch_rechub.models.matching import GRU4Rec

# 定义序列特征
user_features = [
    SequenceFeature(name="user_history", vocab_size=100000, embed_dim=32, pooling=None)
]

# 创建模型
model = GRU4Rec(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, 16], "dropout": 0.2},
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | None |
| history_features | list | `pooling="concat"` 的历史序列；当前实现只使用第一个序列 | 必填 |
| item_features | list | 正样本物品 ID 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 ID 序列 | 必填 |
| user_params | dict | 用户 MLP 参数，最后一层需与 item embedding 维度一致 | 必填 |
| temperature | float | 温度参数 | 0.02 |

> 当前 `GRU4Rec.forward()` 的 list-wise 求和维度与候选物品轴不一致，不建议把这一节的构造片段当作稳定的可训练教程；待实现修正后应配合 list-wise 数据使用。

### 适用场景

- 会话推荐
- 短序列推荐
- 电商场景

## 7. NARM

### 功能描述

NARM（Neural Attentive Session-based Recommendation）是一种基于注意力机制的会话推荐模型，能够同时捕获会话内的局部和全局兴趣。

### 论文引用

```
Li, Jing, et al. "Neural attentive session-based recommendation." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.
```

### 核心原理

- **GRU 序列建模**：使用 GRU 捕捉会话内的序列依赖
- **注意力机制**：引入注意力机制，捕获会话内的局部兴趣
- **全局表示**：同时学习会话的全局表示，结合局部和全局兴趣
- **会话推荐**：专注于会话内的推荐

### 使用方法

```python
from torch_rechub.models.matching import NARM

# 创建模型
model = NARM(
    item_history_feature=item_history_feature,
    hidden_dim=64,
    emb_dropout_p=0.2,
    session_rep_dropout_p=0.2,
    item_feature=item_feature,  # 仅在 batch 内负采样/物品塔推理时需要
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | 会话 item 序列，0 为 padding | 必填 |
| hidden_dim | int | GRU 隐藏维度 | 必填 |
| emb_dropout_p | float | item embedding dropout | 必填 |
| session_rep_dropout_p | float | 会话表示 dropout | 必填 |
| item_feature | SparseFeature or None | 可选的目标 item ID，用于独立 item tower/批内负采样 | None |

### 适用场景

- 会话推荐
- 短序列推荐
- 局部和全局兴趣建模

## 8. SASRec

### 功能描述

SASRec（Self-Attentive Sequential Recommendation）是一种基于自注意力机制的序列推荐模型，能够捕获用户行为序列中的长距离依赖关系。

### 论文引用

```
Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.
```

### 核心原理

- **自注意力机制**：使用多头自注意力机制捕获序列中的长距离依赖
- **位置编码**：添加位置信息，保留序列的顺序信息
- **层归一化**：加速模型收敛，提高训练稳定性
- **残差连接**：增强模型的表达能力

### 使用方法

```python
from torch_rechub.models.matching import SASRec

# 创建模型
model = SASRec(
    features=[seq_feature, pos_feature, neg_feature],
    max_len=50,
    dropout_rate=0.2,
    num_blocks=2,
    num_heads=4,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 顺序固定为历史序列、正样本序列、负样本序列，同为 `SequenceFeature` | 必填 |
| max_len | int | 序列长度 | 50 |
| dropout_rate | float | Embedding/注意力/FFN dropout | 0.5 |
| num_blocks | int | Self-Attention block 数 | 2 |
| num_heads | int | 注意力头数，需整除 embedding 维度 | 1 |
| item_feature | SparseFeature or None | 实验性的独立 item tower 特征 | None |

> 当前 SASRec 在 `forward` 中构造 CPU mask/position，而 `item_feature` 分支还访问了不存在的 embedding 属性；因此本页只列出源码中的真实构造接口，不将 GPU 或 item-tower 路径声明为可用。

### 适用场景

- 长序列推荐
- 用户行为序列重要的场景
- 顺序推荐任务

## 9. SINE

### 功能描述

SINE（Sparse Interest Network for Sequential Recommendation）是一种稀疏兴趣网络，能够有效地建模用户的稀疏兴趣。

### 论文引用

```
Chen, Jiaxi, et al. "SINE: A sparse interest network for sequential recommendation." Proceedings of the 14th ACM Conference on Recommender Systems. 2021.
```

### 核心原理

- **稀疏兴趣建模**：专门处理用户兴趣稀疏的问题
- **动态路由**：使用动态路由机制，自适应地聚合用户兴趣
- **兴趣演化**：捕捉用户兴趣的动态变化
- **高效计算**：优化了计算效率，适合大规模数据

### 使用方法

```python
from torch_rechub.models.matching import SINE

# 创建模型
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

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| history_features | list[str] | 历史 item 序列字段名；当前只使用第一个 | 必填 |
| item_features | list[str] | 正样本 item 字段名 | 必填 |
| neg_item_features | list[str] | 负样本 item 序列字段名 | 必填 |
| num_items | int | 包含 padding 的 item 词表大小 | 必填 |
| embedding_dim | int | Embedding 维度 | 必填 |
| hidden_dim | int | 注意力隐藏维度 | 必填 |
| num_concept | int | 全局概念原型数 | 必填 |
| num_intention | int | 每个用户选取的意图数 | 必填 |
| seq_max_len | int | 输入固定序列长度 | 必填 |
| num_heads | int | 自注意力头数 | 1 |
| temperature | float | 温度参数 | 0.02 |

### 适用场景

- 用户兴趣稀疏的场景
- 大规模推荐系统
- 电商推荐

## 10. STAMP

### 功能描述

STAMP（Short-Term Attention/Memory Priority Model）是一种基于注意力机制的会话推荐模型，专注于近期用户行为的建模。

### 论文引用

```
Liu, Qiao, et al. "STAMP: short-term attention/memory priority model for session-based recommendation." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

### 核心原理

- **短期注意力**：专注于近期用户行为，给予近期行为更高的权重
- **记忆模块**：维护一个记忆向量，捕获会话的全局信息
- **会话推荐**：专注于会话内的推荐
- **简单高效**：模型结构简单，计算效率高

### 使用方法

```python
from torch_rechub.models.matching import STAMP

# 创建模型
model = STAMP(
    item_history_feature=item_history_feature,
    weight_std=0.05,
    emb_std=0.05,
    item_feature=item_feature,  # 仅在 batch 内负采样/物品塔推理时需要
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | 会话 item 序列，0 为 padding | 必填 |
| weight_std | float | 注意力参数初始化标准差 | 必填 |
| emb_std | float | Embedding/线性层初始化标准差 | 必填 |
| item_feature | SparseFeature or None | 可选的目标 item ID，用于 item tower/批内负采样 | None |

### 适用场景

- 会话推荐
- 短期兴趣建模
- 电商场景

## 11. ComirecDR

### 功能描述

ComirecDR（Controllable Multi-Interest Recommendation with Dynamic Routing）是一种可控多兴趣推荐模型，允许控制生成的兴趣数量。

### 论文引用

```
Chen, Jiaxi, et al. "Controllable multi-interest framework for recommendation." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
```

### 核心原理

- **可控多兴趣**：允许控制生成的兴趣数量
- **动态路由**：使用动态路由机制，自适应地聚合用户兴趣
- **双塔结构**：支持离线预计算物品嵌入
- **高效计算**：优化了计算效率，适合大规模数据

### 使用方法

```python
from torch_rechub.models.matching import ComirecDR

# 创建模型
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

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 非序列用户特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史 item 序列 | 必填 |
| item_features | list | 正样本 item 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 item 序列 | 必填 |
| max_length | int | 固定历史序列长度 | 必填 |
| interest_num | int | 兴趣数量 | 4 |
| temperature | float | 温度参数 | 0.02 |

### 适用场景

- 可控多兴趣推荐
- 用户兴趣多样化的场景
- 大规模推荐系统

## 12. ComirecSA

### 功能描述

ComirecSA（Controllable Multi-Interest Recommendation with Self-Attention）是 Comirec 的自注意力版本，使用自注意力机制建模用户兴趣。

### 核心原理

- **自注意力机制**：使用自注意力机制捕获用户行为序列中的依赖关系
- **可控多兴趣**：允许控制生成的兴趣数量
- **双塔结构**：支持离线预计算物品嵌入
- **高效计算**：优化了计算效率，适合大规模数据

### 使用方法

```python
from torch_rechub.models.matching import ComirecSA

# 创建模型
model = ComirecSA(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    interest_num=4,
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 非序列用户特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史 item 序列 | 必填 |
| item_features | list | 正样本 item 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 item 序列 | 必填 |
| interest_num | int | 兴趣数量 | 4 |
| temperature | float | 温度参数 | 0.02 |

### 适用场景

- 可控多兴趣推荐
- 长序列兴趣建模
- 大规模推荐系统

## 13. 模型比较

| 模型 | 复杂度 | 表达能力 | 计算效率 | 适用场景 |
| --- | --- | --- | --- | --- |
| DSSM | 低 | 中 | 高 | 文本匹配、冷启动 |
| FaceBookDSSM | 中 | 高 | 中 | 大规模推荐、广告召回 |
| YoutubeDNN | 中 | 高 | 中 | 视频推荐、内容推荐 |
| YoutubeSBC | 中 | 高 | 中 | 大规模推荐、采样偏差场景 |
| MIND | 中 | 高 | 中 | 多兴趣推荐、电商场景 |
| GRU4Rec | 中 | 中 | 高 | 会话推荐、短序列推荐 |
| NARM | 中 | 中 | 中 | 会话推荐、局部全局兴趣 |
| SASRec | 高 | 高 | 低 | 长序列推荐、顺序推荐 |
| SINE | 高 | 高 | 中 | 稀疏兴趣建模、大规模推荐 |
| STAMP | 低 | 中 | 高 | 会话推荐、短期兴趣 |
| ComirecDR | 中 | 高 | 中 | 可控多兴趣、大规模推荐 |
| ComirecSA | 高 | 高 | 中 | 可控多兴趣、长序列推荐 |

## 14. 使用建议

1. **根据数据特点选择模型**：
   - 长序列数据推荐使用 SASRec、ComirecSA
   - 会话数据推荐使用 GRU4Rec、NARM、STAMP
   - 多兴趣场景推荐使用 MIND、ComirecDR、ComirecSA

2. **根据计算资源选择模型**：
   - 计算资源有限时推荐使用 DSSM、GRU4Rec、STAMP
   - 计算资源充足时可以尝试更复杂的模型，如 SASRec、SINE

3. **考虑业务需求**：
   - 需要可控多兴趣时推荐使用 ComirecDR、ComirecSA
   - 需要处理稀疏兴趣时推荐使用 SINE
   - 需要处理采样偏差时推荐使用 YoutubeSBC

4. **尝试多种召回策略融合**：
   - 不同召回模型可能捕获不同的用户兴趣，模型融合可以提高最终召回效果
   - 可以结合内容召回、协同召回等多种召回策略

## 15. 代码示例：完整的召回模型训练流程

```python
import os

from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 1. 定义特征
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1),
    SparseFeature(name="gender", vocab_size=3, embed_dim=8)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# 2. 准备数据
# 假设 x 和 y 是已经处理好的特征和标签数据
# x 包含用户特征和物品特征
x = {
    "user_id": user_id_data,
    "age": age_data,
    "gender": gender_data,
    "item_id": item_id_data,
    "category": category_data
}
y = label_data  # 点击/不点击标签

# 测试用户数据和所有物品数据
x_test_user = {
    "user_id": test_user_id_data,
    "age": test_age_data,
    "gender": test_gender_data
}
x_all_item = {
    "item_id": all_item_id_data,
    "category": all_item_category_data
}

# 3. 创建数据生成器
dg = MatchDataGenerator(x, y)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    x_test_user=x_test_user,
    x_all_item=x_all_item,
    batch_size=256,
    num_workers=8
)

# 4. 创建模型
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # 当前 DSSM forward 尚未使用该参数
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"}
)

# 5. 创建训练器
trainer = MatchTrainer(
    model=model,
    mode=0,  # 0: point-wise, 1: pair-wise, 2: list-wise
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cpu",
    model_path="saved/dssm"
)

# 6. 训练模型
os.makedirs("saved/dssm", exist_ok=True)
# test_dl / item_dl 是无标签的向量推理 DataLoader，不能作为验证集传给 fit
trainer.fit(train_dl)

# 7. 导出 ONNX 模型（先安装：pip install "torch-rechub[onnx]"）
# 导出用户塔
trainer.export_onnx("user_tower.onnx", mode="user")
# 导出物品塔
trainer.export_onnx("item_tower.onnx", mode="item")

# 8. 向量召回示例
# 生成用户嵌入
user_embeddings = trainer.inference_embedding(model, mode="user", data_loader=test_dl, model_path="saved/dssm")
# 生成物品嵌入
item_embeddings = trainer.inference_embedding(model, mode="item", data_loader=item_dl, model_path="saved/dssm")

# 使用 Annoy 或 Faiss 进行向量索引和召回
# 这里以 Annoy 为例
# 先安装：pip install "torch-rechub[annoy]"
from annoy import AnnoyIndex

# 创建索引
index = AnnoyIndex(64, 'angular')  # 64 是嵌入维度
for i, embedding in enumerate(item_embeddings):
    index.add_item(i, embedding.tolist())
index.build(10)  # 10棵树

# 召回示例
user_idx = 0
user_emb = user_embeddings[user_idx].tolist()
recall_results = index.get_nns_by_vector(user_emb, 10)  # 召回 top 10 物品
print(f"User {user_idx} 召回的物品: {recall_results}")
```

## 16. 常见问题与解决方案

### Q: 如何处理大规模物品集？
A: 可以尝试以下方法：
- 使用双塔结构，支持离线预计算物品嵌入
- 使用近似最近邻搜索库（如 Annoy、Faiss）加速向量检索
- 采用分层召回策略，先粗召回再精召回

### Q: 如何处理冷启动问题？
A: 可以尝试以下方法：
- 对于新用户，使用基于内容的召回
- 对于新物品，使用协同过滤或基于内容的召回
- 使用迁移学习，从其他相关领域迁移知识

### Q: 如何评估召回模型的效果？
A: 常用的召回评估指标包括：
- Recall@K：前 K 个召回结果中相关物品的比例
- Precision@K：前 K 个召回结果中相关物品的比例
- NDCG@K：考虑排序的召回质量
- Hit@K：是否召回至少一个相关物品
- MRR@K：平均倒数排名

### Q: 如何选择合适的负采样策略？
A: 常见的负采样策略包括：
- 随机负采样：简单高效，但可能采样到不相关的物品
- 基于流行度的负采样：根据物品流行度采样，更符合实际情况
- 困难负采样：采样与正样本相似的负样本，提高模型的区分能力
- 对比学习中的负采样：如 MoCo、SimCLR 等方法

### Q: 如何优化召回模型的性能？
A: 可以尝试以下方法：
- 增加模型深度和宽度，提高模型表达能力
- 使用更高级的特征工程
- 优化负采样策略
- 尝试模型融合
- 调整温度参数，优化相似度分布

## 17. 召回模型部署建议

1. **离线预计算**：对于双塔模型，离线预计算物品嵌入，减少在线计算量
2. **向量索引**：使用高效的向量索引库（如 Faiss）存储物品嵌入，加速在线检索
3. **分层部署**：采用分层召回架构，先使用简单模型粗召回，再使用复杂模型精召回
4. **缓存机制**：缓存高频用户的召回结果，减少重复计算
5. **实时更新**：定期更新物品嵌入，保证召回结果的时效性
6. **A/B 测试**：通过 A/B 测试验证不同召回策略的效果

召回模型是推荐系统的重要组成部分，选择合适的召回模型并进行优化，可以显著提高推荐系统的整体效果。Torch-RecHub 提供了丰富的召回模型，涵盖了不同的召回策略，方便开发者根据业务需求选择和使用。
