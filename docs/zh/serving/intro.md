---
title: 生产部署导览
description: Torch-RecHub 模型部署指南
---

# 生产部署导览

Torch-RecHub 提供 ONNX 导出、量化和向量索引等部署组件。生产环境仍需自行补齐特征服务、API 服务、监控、灰度与容灾等工程能力。

## 部署流程概览

![生产部署全流程图](/img/diagrams/serving_pipeline.png)

## 核心功能

| 功能 | 描述 | 文档链接 |
| --- | --- | --- |
| **ONNX 导出** | 将 PyTorch 模型导出为 ONNX 格式 | [ONNX 导出与量化](/zh/serving/onnx) |
| **模型量化** | INT8/FP16 量化，降低推理延迟 | [ONNX 导出与量化](/zh/serving/onnx) |
| **向量检索** | Annoy/FAISS/Milvus 向量索引 | [向量检索封装](/zh/serving/vector_index) |
| **在线服务** | 部署示例和最佳实践 | [在线服务示例](/zh/serving/demo) |

## 快速开始

### 1. ONNX 导出

```python
from torch_rechub.trainers import CTRTrainer, MatchTrainer

# 排序模型训练完成后导出
ctr_trainer = CTRTrainer(ctr_model)
ctr_trainer.export_onnx("model.onnx")

# 双塔模型分别导出
match_trainer = MatchTrainer(match_model)
match_trainer.export_onnx("user_tower.onnx", mode="user")
match_trainer.export_onnx("item_tower.onnx", mode="item")
```

### 2. 模型量化

```python
from torch_rechub.utils.quantization import quantize_model

# INT8 量化（推荐 CPU）
quantize_model("model_fp32.onnx", "model_int8.onnx", mode="int8")

# FP16 量化（推荐 GPU）
quantize_model("model_fp32.onnx", "model_fp16.onnx", mode="fp16")
```

### 3. 向量检索

当前 `torch_rechub.serving` 会在导入时加载 Annoy、FAISS 和 Milvus 三个后端；使用统一工厂前请安装全部检索 extra：

```bash
pip install "torch-rechub[annoy,faiss,milvus]"
```

```python
from torch_rechub.serving import builder_factory

# 创建 FAISS 索引
builder = builder_factory("faiss", index_type="HNSW", metric="IP")

with builder.from_embeddings(item_embeddings) as indexer:
    ids, scores = indexer.query(user_embeddings, top_k=10)
    indexer.save("item.index")
```

## 部署架构建议

### 排序模型部署

```
用户请求 → 特征服务 → ONNX Runtime → 排序结果
```

### 召回模型部署

```
用户请求 → 用户塔推理 → 向量检索 → 召回结果
                ↓
        物品塔离线计算 → 向量索引
```

## 下一步

- 了解 [ONNX 导出与量化](/zh/serving/onnx) 的详细用法
- 了解 [向量检索封装](/zh/serving/vector_index) 的配置方法
- 查看 [在线服务示例](/zh/serving/demo) 了解完整部署流程

