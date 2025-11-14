# Embedding正则化使用示例

## 基本用法

### 1. 定义特征时指定正则化参数

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature, DenseFeature

# 定义特征，指定L1和L2正则化参数
features = [
    # L1和L2正则化
    SparseFeature("user_id", vocab_size=1000, embed_dim=16, 
                  l1_reg=0.001, l2_reg=0.0001),
    
    # 只指定L1正则化
    SparseFeature("item_id", vocab_size=5000, embed_dim=16, 
                  l1_reg=0.001),
    
    # 只指定L2正则化
    SparseFeature("category", vocab_size=100, embed_dim=16, 
                  l2_reg=0.0001),
    
    # 不指定正则化（默认）
    SparseFeature("brand", vocab_size=500, embed_dim=16),
    
    # SequenceFeature也支持正则化
    SequenceFeature("hist_item_id", vocab_size=5000, embed_dim=16,
                    pooling="mean", shared_with="item_id",
                    l1_reg=0.001, l2_reg=0.0001),
    
    # DenseFeature不需要正则化
    DenseFeature("price", embed_dim=1),
]
```

### 2. 创建模型（无需修改）

```python
from torch_rechub.models.ranking import DeepFM

model = DeepFM(
    features=features,
    embed_dim=16,
    mlp_params={"dims": [256, 128], "dropout": 0.5}
)
```

### 3. 创建Trainer并训练（无需修改）

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(
    model=model,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)

# 正则化损失会自动添加到总损失中
trainer.fit(train_loader, val_loader)
```

## 工作原理

1. **特征定义阶段**
   - 在SparseFeature/SequenceFeature中指定l1_reg和l2_reg参数
   - 参数默认为None，会转换为0.0（不使用正则化）

2. **EmbeddingLayer初始化**
   - 创建embedding时，将l1_reg和l2_reg存储在reg_dict中
   - 格式: `{embed_name: (l1_coef, l2_coef)}`

3. **训练过程**
   - 每个batch计算正则化损失: `reg_loss = embedding_layer.get_regularization_loss()`
   - 正则化损失自动添加到总损失中
   - 反向传播时自动计算梯度

## 正则化损失计算

```python
# L1正则化: sum(|w|)
l1_loss = l1_coef * torch.norm(embed.weight, p=1)

# L2正则化: sqrt(sum(w^2))
l2_loss = l2_coef * torch.norm(embed.weight, p=2)

# 总正则化损失
reg_loss = l1_loss + l2_loss
```

## 向后兼容性

✅ 现有代码无需修改
- 不指定l1_reg和l2_reg时，默认为0.0（不使用正则化）
- 现有模型和trainer无需修改
- 完全向后兼容

## 支持的模型

所有ranking、matching和multi-task模型都自动支持：
- Ranking: DeepFM, DCN, DCNv2, EDCN, AFM, FiBiNet, WideDeep, DIN, DIEN, BST等
- Matching: DSSM, YoutubeDNN, MIND, GRU4Rec, SASRec等
- Multi-task: MMOE, PLE, SharedBottom, ESMM, AITM等

## 常见问题

**Q: 如何调整正则化强度？**
A: 修改l1_reg和l2_reg的值。通常范围是0.0001到0.01。

**Q: 如何只使用L1或L2正则化？**
A: 只指定需要的参数，另一个参数不指定或设为None。

**Q: 正则化损失如何影响训练？**
A: 正则化损失会自动添加到总损失中，鼓励embedding权重更小，防止过拟合。

