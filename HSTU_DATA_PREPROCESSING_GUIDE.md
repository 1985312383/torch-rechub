# HSTU模型 - MovieLens-1M数据预处理指南

## 📋 概述

本指南说明如何使用MovieLens-1M数据集来训练HSTU模型。

---

## 📁 文件结构

```
examples/generative/
├── data/
│   └── ml-1m/
│       ├── preprocess_ml_hstu.py      # 数据预处理脚本
│       ├── ml-1m_sample.csv           # 样本数据
│       └── processed/                 # 处理后的数据（自动生成）
│           ├── train_data.pkl
│           ├── val_data.pkl
│           ├── test_data.pkl
│           └── vocab.pkl
└── run_hstu_movielens.py              # 训练脚本
```

---

## 🚀 快速开始

### 步骤1：准备原始数据

从MovieLens官网下载ml-1m数据集：
- 下载链接: https://grouplens.org/datasets/movielens/1m/
- 解压到 `examples/generative/data/ml-1m/` 目录

目录结构应该是：
```
examples/generative/data/ml-1m/
├── ratings.dat
├── movies.dat
├── users.dat
└── README
```

### 步骤2：运行数据预处理

```bash
cd examples/generative/data/ml-1m/
python preprocess_ml_hstu.py
```

这将生成以下文件：
- `processed/train_data.pkl` - 训练数据
- `processed/val_data.pkl` - 验证数据
- `processed/test_data.pkl` - 测试数据
- `processed/vocab.pkl` - 词表映射

### 步骤3：训练模型

使用真实数据训练：
```bash
cd examples/generative/
python run_hstu_movielens.py --use-real-data
```

或使用虚拟数据测试：
```bash
python run_hstu_movielens.py
```

---

## 📊 数据预处理详解

### MovieLensHSTUPreprocessor类

#### 初始化参数

```python
preprocessor = MovieLensHSTUPreprocessor(
    data_dir="./data/ml-1m/",           # 原始数据目录
    output_dir="./data/ml-1m/processed/" # 输出目录
)
```

#### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| min_seq_len | 5 | 最小序列长度 |
| max_seq_len | 256 | 最大序列长度 |
| rating_threshold | 3 | 评分阈值（>=3为正样本） |

#### 处理流程

1. **加载数据**: 读取ratings.dat和movies.dat
2. **过滤数据**: 只保留评分>=3的交互
3. **构建序列**: 按用户和时间构建行为序列
4. **创建词表**: 映射movie_id到token_id
5. **生成样本**: 为每个用户生成(历史序列, 目标item)对
6. **分割数据**: 按7:1:2比例分割train/val/test
7. **保存数据**: 保存为pickle文件

---

## 📈 数据格式

### 输入格式

每个样本包含：
- `seq_tokens`: [max_seq_len] - 序列中的item token
- `seq_positions`: [max_seq_len] - 位置编码
- `targets`: 标量 - 目标item的token

### 数据统计

```
原始交互数: ~1,000,209
有效用户数: ~6,040
有效电影数: ~3,706
训练样本数: ~4,228
验证样本数: ~605
测试样本数: ~605
```

---

## 🔧 自定义配置

### 修改序列长度

编辑 `preprocess_ml_hstu.py`:

```python
preprocessor = MovieLensHSTUPreprocessor()
preprocessor.max_seq_len = 512  # 改为512
preprocessor.preprocess()
```

### 修改评分阈值

```python
preprocessor.rating_threshold = 4  # 只保留评分>=4的
```

### 修改数据分割比例

编辑 `preprocess_ml_hstu.py` 中的 `split_data` 方法：

```python
train_ratio = 0.8  # 80%训练
val_ratio = 0.1    # 10%验证
# 剩余10%为测试
```

---

## 💾 数据加载

### 使用SequenceDataGenerator

```python
from torch_rechub.utils.data import SequenceDataGenerator
import pickle

# 加载数据
with open('processed/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

# 创建数据加载器
data_gen = SequenceDataGenerator(
    train_data['seq_tokens'],
    train_data['seq_positions'],
    train_data['targets']
)

train_loader = data_gen.generate_dataloader(batch_size=32)[0]
```

### 直接使用numpy数组

```python
import pickle

with open('processed/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

seq_tokens = train_data['seq_tokens']      # [N, 256]
seq_positions = train_data['seq_positions'] # [N, 256]
targets = train_data['targets']             # [N]
```

---

## 🎯 常见问题

### Q: 数据预处理需要多长时间？
A: 取决于硬件，通常5-10分钟。

### Q: 如何处理缺失的电影？
A: 预处理脚本会自动过滤不存在的电影。

### Q: 可以修改序列长度吗？
A: 可以，修改 `max_seq_len` 参数即可。

### Q: 如何添加其他特征？
A: 修改 `generate_training_data` 方法以包含额外特征。

---

## 📝 输出示例

预处理完成后的输出：

```
============================================================
MovieLens-1M HSTU数据预处理
============================================================
加载MovieLens-1M数据...
加载完成: 1000209 条交互记录
构建用户序列...
有效用户数: 6040
创建词表...
词表大小: 3707
生成训练数据...
生成数据: 4438 条样本
分割数据集...
Train: 3106, Val: 444, Test: 888
保存数据...
保存 train 数据到 ./data/ml-1m/processed/train_data.pkl
保存 val 数据到 ./data/ml-1m/processed/val_data.pkl
保存 test 数据到 ./data/ml-1m/processed/test_data.pkl
保存词表到 ./data/ml-1m/processed/vocab.pkl
============================================================
预处理完成！
============================================================
```

---

## 🔗 相关文件

- `preprocess_ml_hstu.py` - 数据预处理脚本
- `run_hstu_movielens.py` - 训练脚本
- `HSTU_QUICK_START.md` - 快速开始指南
- `HSTU_TECHNICAL_DETAILS.md` - 技术细节

---

**版本**: v1.0  
**日期**: 2025-11-14  
**状态**: ✅ 完成

