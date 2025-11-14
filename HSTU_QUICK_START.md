# HSTU模型 - 快速开始指南

## 🚀 5分钟快速开始

### 1. 导入必要的模块

```python
import torch
import numpy as np
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.utils.data import SequenceDataGenerator
from torch_rechub.trainers.seq_trainer import SeqTrainer
```

### 2. 准备数据

```python
# 生成示例数据
num_samples = 1000
seq_len = 256
vocab_size = 10000

seq_tokens = np.random.randint(2, vocab_size, size=(num_samples, seq_len))
seq_positions = np.tile(np.arange(seq_len), (num_samples, 1))
targets = np.random.randint(2, vocab_size, size=(num_samples,))

# 创建数据加载器
data_gen = SequenceDataGenerator(seq_tokens, seq_positions, targets)
train_loader, val_loader, test_loader = data_gen.generate_dataloader(
    batch_size=32,
    split_ratio=(0.7, 0.1, 0.2)
)
```

### 3. 创建模型

```python
model = HSTUModel(
    vocab_size=vocab_size,
    d_model=256,
    n_heads=8,
    n_layers=2,
    max_seq_len=256,
    dropout=0.1
)
```

### 4. 训练模型

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = SeqTrainer(model, optimizer, device='cuda')

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    early_stopping_patience=3,
    save_path='hstu_model.pt'
)
```

### 5. 评估模型

```python
test_loss, test_accuracy = trainer.evaluate(test_loader)
print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy:.4f}")
```

---

## 📚 完整示例

运行完整示例：

```bash
python examples/generative/run_hstu_movielens.py
```

---

## 🔧 常见配置

### 小模型（快速测试）
```python
model = HSTUModel(
    vocab_size=10000,
    d_model=128,
    n_heads=4,
    n_layers=2,
    max_seq_len=128
)
```

### 中等模型（推荐）
```python
model = HSTUModel(
    vocab_size=100000,
    d_model=256,
    n_heads=8,
    n_layers=4,
    max_seq_len=256
)
```

### 大模型（高性能）
```python
model = HSTUModel(
    vocab_size=1000000,
    d_model=512,
    n_heads=16,
    n_layers=6,
    max_seq_len=512
)
```

---

## 📊 参数说明

| 参数 | 说明 | 默认值 | 范围 |
|------|------|--------|------|
| vocab_size | 词表大小 | - | > 0 |
| d_model | 模型维度 | 512 | 128-1024 |
| n_heads | 多头数 | 8 | 4-16 |
| n_layers | 层数 | 4 | 2-12 |
| dqk | Query/Key维度 | 64 | 32-128 |
| dv | Value维度 | 64 | 32-128 |
| max_seq_len | 最大序列长度 | 256 | 64-1024 |
| dropout | Dropout概率 | 0.1 | 0.0-0.5 |

---

## 🎯 关键类和方法

### HSTUModel
```python
model = HSTUModel(vocab_size=10000)
logits = model(x)  # (B, L) -> (B, L, V)
```

### SequenceDataGenerator
```python
gen = SequenceDataGenerator(seq_tokens, seq_positions, targets)
train_loader, val_loader, test_loader = gen.generate_dataloader(batch_size=32)
```

### SeqTrainer
```python
trainer = SeqTrainer(model, optimizer, device='cuda')
history = trainer.fit(train_loader, val_loader, epochs=20)
test_loss, accuracy = trainer.evaluate(test_loader)
```

---

## 💡 最佳实践

### 1. 数据准备
- 确保seq_tokens、seq_positions、targets长度一致
- 使用合适的batch_size（通常32-128）
- 考虑数据的train/val/test分割比例

### 2. 模型配置
- 根据显存选择合适的d_model和n_layers
- 使用n_heads = d_model / dqk
- 调整dropout以防止过拟合

### 3. 训练策略
- 使用Adam优化器，学习率0.001-0.0001
- 启用早停以防止过拟合
- 定期保存最佳模型

### 4. 评估指标
- 关注验证集损失和准确率
- 考虑添加Recall@k等指标
- 在测试集上进行最终评估

---

## 🐛 常见问题

### Q: 显存不足怎么办？
A: 减小batch_size、d_model或max_seq_len

### Q: 训练速度太慢？
A: 增加num_workers、使用GPU、减少n_layers

### Q: 模型准确率不高？
A: 增加训练轮数、调整学习率、增加模型容量

### Q: 如何使用自己的数据？
A: 将数据转换为seq_tokens、seq_positions、targets的numpy数组格式

---

## 📖 更多资源

- **详细文档**: HSTU_MODIFICATION_GUIDE.md
- **API设计**: HSTU_API_DESIGN_VERIFICATION.md
- **技术细节**: HSTU_TECHNICAL_DETAILS.md
- **实现完成**: HSTU_IMPLEMENTATION_COMPLETE.md

---

## ✅ 检查清单

- [ ] 已安装torch-rechub
- [ ] 已准备好数据
- [ ] 已创建模型
- [ ] 已配置优化器
- [ ] 已开始训练
- [ ] 已评估模型

---

**版本**: v1.0  
**日期**: 2025-11-14  
**状态**: ✅ 可用

