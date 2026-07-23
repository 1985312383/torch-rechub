---
title: 数据流水线
description: Torch-RecHub 数据加载与预处理
---

# 数据流水线

Torch-RecHub 提供常用的数据集类、DataLoader 生成器和预处理工具。这些类负责把**已经完成编码和补齐**的数据交给模型；类别词表、缺失值和数据切分策略仍需使用者根据数据集处理。

![数据流水线图](/img/diagrams/data_pipeline.png)

## 数据类

### TorchDataset

用于训练和验证的数据集合，包含特征和标签。

```python
from torch_rechub.utils.data import TorchDataset

# x 可以是特征名到数组/Tensor 的映射，也可以是 DataFrame
dataset = TorchDataset(x, y)
```

**参数说明：**
- `x`：支持 `.items()` 且每列可按行索引的数据，通常是特征字典或 `pandas.DataFrame`
- `y`：标签数据

### PredictDataset

用于预测的数据集合，仅包含特征。

```python
from torch_rechub.utils.data import PredictDataset

# 创建预测数据集
dataset = PredictDataset(x)
```

**参数说明：**
- `x`：特征字典，键为特征名称，值为特征数据

## 数据生成器

### DataGenerator

用于生成排序模型和多任务模型的数据加载器。

```python
from torch_rechub.utils.data import DataGenerator

# 创建数据生成器
dg = DataGenerator(x, y)
# 生成数据加载器
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],  # 70% 训练、10% 验证，剩余 20% 测试
    batch_size=256,           # 批次大小
    num_workers=8             # 并行工作线程数
)
```

**参数说明：**
- `x`：特征数据
- `y`：标签数据

**generate_dataloader方法参数：**
- `split_ratio`：长度为 2，分别是训练集和验证集比例；测试集使用剩余数据
- `batch_size`：批次大小
- `num_workers`：并行工作线程数

如果数据已经分好，不要传 `split_ratio`，而是传入 `x_val, y_val, x_test, y_test`：

```python
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=256,
)
```

> `split_ratio` 路径底层使用 `torch.utils.data.random_split`。如需可复现切分，请在生成 DataLoader 之前设置 `torch.manual_seed(...)`。

### MatchDataGenerator

用于生成召回模型的数据加载器。

```python
from torch_rechub.utils.data import MatchDataGenerator

# 创建召回数据生成器
dg = MatchDataGenerator(x, y)
# 生成数据加载器
train_dl, test_dl, item_dl = dg.generate_dataloader(
    x_test_user=x_test_user,  # 测试用户数据
    x_all_item=x_all_item,    # 所有物品数据
    batch_size=256,           # 批次大小
    num_workers=8             # 并行工作线程数
)
```

**参数说明：**
- `x`：特征数据
- `y`：标签数据，可选

**generate_dataloader方法参数：**
- `x_test_user`：测试用户数据
- `x_all_item`：所有物品数据
- `batch_size`：批次大小
- `num_workers`：并行工作线程数

## 工具函数

### get_auto_embedding_dim

根据类别数量自动计算嵌入向量长度。

```python
from torch_rechub.utils.data import get_auto_embedding_dim

# 自动计算嵌入向量长度
embed_dim = get_auto_embedding_dim(num_classes=1000)
```

**参数说明：**
- `num_classes`：类别数量

**返回值：**
- 嵌入向量长度，计算公式：`int(np.floor(6 * np.pow(num_classes, 0.25)))`

### get_loss_func

根据任务类型获取对应的损失函数。

```python
from torch_rechub.utils.data import get_loss_func

# 获取分类任务损失函数
loss_func = get_loss_func(task_type="classification")
# 获取回归任务损失函数
loss_func = get_loss_func(task_type="regression")
```

**参数说明：**
- `task_type`：任务类型，可选值：classification（分类）、regression（回归）

**返回值：**
- `classification` 返回 `torch.nn.BCELoss`（输入应是 `[0, 1]` 内的概率，不是未经 sigmoid 的 logits）
- `regression` 返回 `torch.nn.MSELoss`

## 序列数据

`SequenceDataGenerator` 面向 HSTU/HLLM 这类 next-item 任务。它接收四个第一维长度一致的 NumPy 数组，每个 batch 返回 `(seq_tokens, seq_positions, seq_time_diffs, targets)`。

```python
from torch_rechub.utils.data import SequenceDataGenerator

generator = SequenceDataGenerator(
    seq_tokens,
    seq_positions,
    targets,
    seq_time_diffs,
)

# 数据已经分好时，返回值是长度为 1 的 tuple
train_dl = generator.generate_dataloader(
    batch_size=32,
    num_workers=0,
)[0]

# 自动切分时必须传入三个且总和为 1 的比例
train_dl, val_dl, test_dl = generator.generate_dataloader(
    batch_size=32,
    split_ratio=(0.7, 0.1, 0.2),
)
```

## Parquet 流式数据加载

在工业界场景中，特征工程通常由 **PySpark** 在大数据集群上完成，数据量可达 GB 到 TB 级别。直接使用 `spark_df.toPandas()` 会导致 Driver OOM。

Torch-RecHub 提供 `ParquetIterableDataset`，支持从 Spark 生成的 Parquet 文件目录流式读取数据，无需将全部数据加载到内存。

### 安装依赖

Parquet 数据加载需要 `bigdata` extra：

```bash
python -m pip install "torch-rechub[bigdata]"
```

### ParquetIterableDataset

继承自 `torch.utils.data.IterableDataset`，支持多进程数据加载。

```python
from torch.utils.data import DataLoader
from torch_rechub.data.dataset import ParquetIterableDataset

# 创建流式数据集
dataset = ParquetIterableDataset(
    ["/data/train1.parquet", "/data/train2.parquet"],
    columns=["user_id", "item_id", "label"],  # 可选，指定读取的列
    batch_size=1024,  # 每批次读取的行数
)

# 创建 DataLoader（注意 batch_size=None）
loader = DataLoader(dataset, batch_size=None, num_workers=4)

# 迭代数据
for batch in loader:
    user_id = batch["user_id"]  # torch.Tensor
    item_id = batch["item_id"]  # torch.Tensor
    label = batch["label"]      # torch.Tensor
```

**参数说明：**
- `file_paths`：Parquet 文件路径列表
- `columns`：要读取的列名列表，`None` 表示读取所有列
- `batch_size`：每批次读取的行数，默认 1024

**特性：**
- **流式读取**：使用 PyArrow Scanner 逐批读取，内存占用恒定
- **多进程支持**：自动将文件分配给不同 worker，避免重复读取
- **类型转换**：自动将 PyArrow 数组转换为 PyTorch Tensor
- **嵌套数组支持**：支持 Spark 的 `Array` 类型列，自动转换为 2D Tensor

### 与 Spark 配合使用

```python
# ========== Spark 端 ==========
# df.write.parquet("/data/train.parquet")

# ========== PyTorch 端 ==========
import glob
from torch.utils.data import DataLoader
from torch_rechub.data.dataset import ParquetIterableDataset

file_paths = glob.glob("/data/train.parquet/*.parquet")
dataset = ParquetIterableDataset(file_paths, batch_size=2048)
loader = DataLoader(dataset, batch_size=None, num_workers=8)
```

### 支持的数据类型

| Parquet/Arrow 类型 | 转换结果 |
|-------------------|---------|
| int8/16/32/64 | torch.float32 |
| float32/64 | torch.float32 |
| boolean | torch.float32 |
| list/array | torch.Tensor (2D) |

> **注意**：嵌套数组（如 Spark 的 `Array<Float>`）要求每行长度相同，否则会抛出 `ValueError`。

> **类型限制**：当前转换器不支持 Arrow 字符串列。请在生成 Parquet 前将类别字符串编码为数值 ID。所有受支持的标量列（包括整数 ID）都会转为 `torch.float32`；传入 Embedding 层时模型会再转为整数索引。`float32` 无法精确表示大于 `2^24` 的所有整数，因此请先将超大业务 ID 重映射到紧凑的连续索引。

## 数据处理流程

1. **特征定义**：使用DenseFeature、SparseFeature、SequenceFeature定义特征
2. **数据加载**：加载原始数据
3. **特征编码**：对类别型特征进行LabelEncoder编码
4. **序列处理**：对序列特征进行填充、截断等处理
5. **样本构造**：构造训练样本，包括负采样等
6. **数据生成**：根据任务使用 DataGenerator、MatchDataGenerator 或 SequenceDataGenerator 生成数据加载器
7. **模型训练**：将数据加载器传入模型进行训练
