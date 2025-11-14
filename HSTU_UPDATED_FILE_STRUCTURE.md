# HSTU模型 - 更新后的文件结构和导入关系

## 📁 完整文件结构

```
torch_rechub/
│
├── basic/
│   ├── __init__.py
│   ├── activation.py
│   ├── features.py
│   ├── initializers.py
│   ├── layers.py ⭐ 修改
│   │   ├── 现有类 (不变)
│   │   │   ├── PredictionLayer
│   │   │   ├── EmbeddingLayer
│   │   │   ├── InputMask
│   │   │   ├── LR
│   │   │   ├── ConcatPooling
│   │   │   ├── AveragePooling
│   │   │   ├── SumPooling
│   │   │   ├── MLP
│   │   │   ├── FM
│   │   │   ├── CIN
│   │   │   ├── CrossLayer
│   │   │   ├── ActivationUnit
│   │   │   ├── DynamicGRU
│   │   │   └── ... (其他现有层)
│   │   │
│   │   └── 新增类 (在文件末尾)
│   │       ├── HSTULayer (核心层)
│   │       └── HSTUBlock (可选)
│   │
│   ├── loss_func.py
│   ├── metric.py
│   └── callback.py
│
├── utils/
│   ├── __init__.py ⭐ 更新
│   ├── data.py ⭐ 修改
│   │   ├── 现有类 (不变)
│   │   │   ├── TorchDataset
│   │   │   ├── PredictDataset
│   │   │   ├── MatchDataGenerator
│   │   │   └── DataGenerator
│   │   │
│   │   ├── 现有函数 (不变)
│   │   │   ├── get_auto_embedding_dim()
│   │   │   ├── get_loss_func()
│   │   │   ├── get_metric_func()
│   │   │   ├── generate_seq_feature()
│   │   │   ├── df_to_dict()
│   │   │   ├── neg_sample()
│   │   │   ├── pad_sequences()
│   │   │   ├── array_replace_with_dict()
│   │   │   └── create_seq_features()
│   │   │
│   │   └── 新增类 (在文件末尾)
│   │       ├── SeqDataset
│   │       └── SequenceDataGenerator
│   │
│   ├── hstu_utils.py ⭐ 新建
│   │   ├── RelPosBias
│   │   ├── VocabMask
│   │   └── VocabMapper
│   │
│   ├── match.py
│   └── mtl.py
│
├── models/
│   ├── __init__.py ⭐ 更新
│   ├── ranking/
│   │   ├── __init__.py
│   │   ├── din.py
│   │   ├── dien.py
│   │   ├── deepfm.py
│   │   └── ... (其他排序模型)
│   │
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── sasrec.py
│   │   ├── gru4rec.py
│   │   ├── narm.py
│   │   └── ... (其他匹配模型)
│   │
│   ├── multi_task/
│   │   ├── __init__.py
│   │   ├── shared_bottom.py
│   │   └── ... (其他多任务模型)
│   │
│   └── generative/ ⭐ 新建目录
│       ├── __init__.py
│       └── hstu.py (HSTUModel主类)
│
├── trainers/
│   ├── __init__.py ⭐ 更新
│   ├── ctr_trainer.py
│   ├── match_trainer.py
│   ├── mtl_trainer.py
│   └── seq_trainer.py ⭐ 新建
│
└── examples/
    ├── ranking/
    │   └── run_ali_ccp_ctr_ranking.py
    │
    ├── matching/
    │   └── run_ml_gru4rec.py
    │
    └── generative/ ⭐ 新建目录
        ├── run_hstu_movielens.py
        └── data/
```

---

## 🔗 导入关系图

### 核心导入路径

```
用户代码
  │
  ├─→ from torch_rechub.basic.layers import HSTULayer
  │
  ├─→ from torch_rechub.utils.data import SequenceDataGenerator
  │
  ├─→ from torch_rechub.utils.hstu_utils import RelPosBias, VocabMask, VocabMapper
  │
  ├─→ from torch_rechub.models.generative.hstu import HSTUModel
  │
  └─→ from torch_rechub.trainers.seq_trainer import SeqTrainer
```

### 模块依赖关系

```
HSTUModel
├─ HSTULayer (from torch_rechub.basic.layers)
├─ RelPosBias (from torch_rechub.utils.hstu_utils)
├─ nn.Embedding
├─ nn.Linear
└─ nn.LayerNorm

SeqTrainer
├─ HSTUModel
├─ nn.CrossEntropyLoss
├─ torch.optim.Optimizer
└─ EarlyStopper

SequenceDataGenerator
├─ SeqDataset
├─ VocabMapper (from torch_rechub.utils.hstu_utils)
├─ DataLoader
└─ pad_sequences (from torch_rechub.utils.data)

SeqDataset
├─ torch.utils.data.Dataset
└─ VocabMapper
```

---

## 📝 导入语句示例

### 基础导入

```python
# 导入层
from torch_rechub.basic.layers import HSTULayer

# 导入数据处理
from torch_rechub.utils.data import SequenceDataGenerator, SeqDataset

# 导入工具
from torch_rechub.utils.hstu_utils import RelPosBias, VocabMask, VocabMapper

# 导入模型
from torch_rechub.models.generative.hstu import HSTUModel

# 导入训练器
from torch_rechub.trainers.seq_trainer import SeqTrainer
```

### 完整示例

```python
import torch
from torch_rechub.basic.layers import HSTULayer
from torch_rechub.utils.data import SequenceDataGenerator
from torch_rechub.utils.hstu_utils import RelPosBias, VocabMapper
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.trainers.seq_trainer import SeqTrainer

# 1. 数据处理
vocab_mapper = VocabMapper(vocab_size=100000)
data_gen = SequenceDataGenerator(
    data=raw_data,
    vocab_mapper=vocab_mapper,
    max_seq_len=256
)
train_loader, val_loader, test_loader = data_gen.generate_dataloader(
    batch_size=32,
    num_workers=4
)

# 2. 构建模型
model = HSTUModel(
    vocab_size=100000,
    d_model=512,
    n_heads=8,
    n_layers=4,
    max_seq_len=256,
    dropout=0.1
)

# 3. 训练
trainer = SeqTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
    device='cuda'
)

trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    early_stopping_patience=3
)
```

---

## 🔄 文件修改清单

### 需要修改的文件

| 文件 | 修改类型 | 修改内容 |
|------|---------|---------|
| torch_rechub/basic/layers.py | 补充 | 添加HSTULayer和HSTUBlock类 |
| torch_rechub/utils/data.py | 补充 | 添加SeqDataset和SequenceDataGenerator类 |
| torch_rechub/models/__init__.py | 更新 | 添加HSTUModel导入 |
| torch_rechub/trainers/__init__.py | 更新 | 添加SeqTrainer导入 |
| torch_rechub/utils/__init__.py | 更新 | 添加hstu_utils导入 |

### 需要新建的文件

| 文件 | 类型 | 内容 |
|------|------|------|
| torch_rechub/utils/hstu_utils.py | 新建 | RelPosBias, VocabMask, VocabMapper |
| torch_rechub/models/generative/__init__.py | 新建 | 导入HSTUModel |
| torch_rechub/models/generative/hstu.py | 新建 | HSTUModel主类 |
| torch_rechub/trainers/seq_trainer.py | 新建 | SeqTrainer类 |
| examples/generative/run_hstu_movielens.py | 新建 | 示例代码 |

---

## ✅ 验证清单

实现完成后需要验证：

- [ ] HSTULayer可以从torch_rechub.basic.layers导入
- [ ] SequenceDataGenerator可以从torch_rechub.utils.data导入
- [ ] RelPosBias等工具可以从torch_rechub.utils.hstu_utils导入
- [ ] HSTUModel可以从torch_rechub.models.generative.hstu导入
- [ ] SeqTrainer可以从torch_rechub.trainers.seq_trainer导入
- [ ] 所有导入都不会产生循环依赖
- [ ] 现有模型和训练器不受影响
- [ ] 示例代码可以正常运行

---

**版本**：v1.0  
**日期**：2025-11-14  
**状态**：待实现

