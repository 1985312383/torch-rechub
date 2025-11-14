# HSTU模型技术细节补充

## 1. HSTULayer详细设计

### 1.1 数学公式

```
输入: X ∈ R^(B×L×d_model)

第1步 - 投影:
proj = f1(X) ∈ R^(B×L×(2*n_heads*dqk + 2*n_heads*dv))

第2步 - 多头分解:
Q = proj[..., :dqk] ∈ R^(B×n_heads×L×dqk)
K = proj[..., dqk:2*dqk] ∈ R^(B×n_heads×L×dqk)
U = proj[..., 2*dqk:2*dqk+dv] ∈ R^(B×n_heads×L×dv)
V = proj[..., 2*dqk+dv:] ∈ R^(B×n_heads×L×dv)

第3步 - 注意力:
logits = Q @ K^T / sqrt(dqk) ∈ R^(B×n_heads×L×L)
logits = logits + rel_pos_bias (可选)
A = softmax(logits, dim=-1)
AV = A @ V ∈ R^(B×n_heads×L×dv)

第4步 - 门控:
AV_flat = reshape(AV) ∈ R^(B×L×(n_heads*dv))
U_flat = reshape(U) ∈ R^(B×L×(n_heads*dv))
gated = LayerNorm(AV_flat) * U_flat

第5步 - 输出投影:
Y = f2(gated) ∈ R^(B×L×d_model)
```

### 1.2 PyTorch实现要点

```python
# 关键操作
1. einsum用于高效张量收缩
   - logits = torch.einsum('bhnd,bhmd->bhnm', Q, K)
   - AV = torch.einsum('bhnm,bhmd->bhnd', A, V)

2. 维度变换
   - permute(0,2,1,3): [B,L,n_heads,d] -> [B,n_heads,L,d]
   - contiguous().view(): 确保内存连续后reshape

3. LayerNorm应用
   - 在门控前对AV进行归一化
   - eps=1e-6防止数值不稳定
```

---

## 2. 相对位置偏置(RelPosBias)

### 2.1 设计方案

```python
class RelPosBias(nn.Module):
    def __init__(self, n_heads, max_seq_len, time_bucket_size=None):
        # n_heads: 多头数
        # max_seq_len: 最大序列长度
        # time_bucket_size: 时间差bucket大小(可选)
        
    def forward(self, seq_len, time_diffs=None):
        # seq_len: 当前序列长度
        # time_diffs: 时间差矩阵[L, L](可选)
        # 返回: [1, n_heads, L, L]
```

### 2.2 实现步骤

```
1. 计算相对距离矩阵
   rel_dist[i,j] = j - i (相对位置)

2. 时间差bucket化(可选)
   如果有时间戳:
   - 计算时间差
   - 按bucket大小离散化
   - 映射到embedding

3. 创建偏置embedding表
   bias_emb: [n_buckets, n_heads]

4. 查表获取偏置
   bias = bias_emb[rel_dist]
   返回形状: [1, n_heads, L, L]
```

### 2.3 时间差bucket化示例

```python
def bucket_time_diff(time_diff, bucket_size=10):
    """
    将时间差离散化为bucket
    例: bucket_size=10
    [0-9] -> bucket 0
    [10-19] -> bucket 1
    ...
    """
    return min(time_diff // bucket_size, max_bucket_id)
```

---

## 3. 序列数据处理

### 3.1 数据格式转换

```
原始日志:
user_id | item_id | action | timestamp
1       | 101     | click  | 1000
1       | 102     | skip   | 1010
1       | 103     | click  | 1020

转换后(交错序列):
seq_tokens = [101, click_id, 102, skip_id, 103, click_id]
seq_positions = [0, 1, 2, 3, 4, 5]
target = 104 (下一个item)

或者(仅item序列):
seq_tokens = [101, 102, 103]
seq_positions = [0, 1, 2]
target = 104
```

### 3.2 Vocab Mapping

```python
class VocabMapper:
    def __init__(self):
        self.item_to_id = {}  # item_id -> token_id
        self.action_to_id = {}  # action -> token_id
        self.id_to_item = {}  # token_id -> item_id
        
    def encode_item(self, item_id):
        return self.item_to_id.get(item_id, UNK_ID)
    
    def decode_item(self, token_id):
        return self.id_to_item.get(token_id, None)
```

### 3.3 Padding和Truncation

```python
def process_sequence(seq, max_len, pad_id=0):
    if len(seq) > max_len:
        # 截断: 保留最后max_len个
        seq = seq[-max_len:]
    else:
        # Padding: 前面补pad_id
        seq = [pad_id] * (max_len - len(seq)) + seq
    return seq
```

---

## 4. 损失函数和评估指标

### 4.1 训练损失

```python
# 交叉熵损失(忽略PAD)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

# 计算
logits = model(seq_tokens, seq_positions)  # [B, L, vocab_size]
last_logits = logits[:, -1, :]  # [B, vocab_size]
loss = criterion(last_logits, target)  # target: [B]
```

### 4.2 评估指标

```python
def top_k_accuracy(logits, target, k=10):
    """Top-k准确率"""
    _, top_k_ids = torch.topk(logits, k, dim=-1)
    correct = (top_k_ids == target.unsqueeze(1)).any(dim=1)
    return correct.float().mean()

def recall_at_k(logits, target, k=10):
    """Recall@k"""
    _, top_k_ids = torch.topk(logits, k, dim=-1)
    correct = (top_k_ids == target.unsqueeze(1)).any(dim=1)
    return correct.float().sum() / len(target)

def mrr(logits, target):
    """Mean Reciprocal Rank"""
    sorted_ids = torch.argsort(logits, descending=True)
    ranks = (sorted_ids == target.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    return (1.0 / ranks).mean()
```

---

## 5. Vocab Mask推理

### 5.1 设计方案

```python
class VocabMask:
    def __init__(self, valid_item_ids):
        # valid_item_ids: 有效item的token ID列表
        self.mask = torch.zeros(vocab_size)
        self.mask[valid_item_ids] = 1.0
    
    def apply(self, logits):
        """
        logits: [B, vocab_size]
        返回: masked logits
        """
        # 将无效item的logits设为-inf
        masked_logits = logits.clone()
        masked_logits[self.mask == 0] = -float('inf')
        return masked_logits
```

### 5.2 推理流程

```python
def inference(model, seq_tokens, seq_positions, vocab_mask=None):
    logits = model(seq_tokens, seq_positions)  # [B, vocab_size]
    
    if vocab_mask is not None:
        logits = vocab_mask.apply(logits)
    
    # 获取top-k候选
    top_k_logits, top_k_ids = torch.topk(logits, k=10)
    
    # 转换回item_id
    item_ids = [vocab_mapper.decode_item(id) for id in top_k_ids]
    
    return item_ids
```

---

## 6. 与框架集成的关键点

### 6.1 特征系统集成

```python
# 不使用框架的复杂特征系统
# 直接使用nn.Embedding

self.item_emb = nn.Embedding(vocab_size, d_model)
self.pos_emb = nn.Embedding(max_seq_len, d_model)

# 前向传播
item_emb = self.item_emb(seq_tokens)  # [B, L, d_model]
pos_emb = self.pos_emb(seq_positions)  # [B, L, d_model]
X = item_emb + pos_emb
```

### 6.2 数据加载器兼容性

```python
# 自定义Dataset
class SeqDataset(Dataset):
    def __init__(self, data_dict):
        self.seq_tokens = data_dict['seq_tokens']
        self.seq_positions = data_dict['seq_positions']
        self.targets = data_dict['targets']
    
    def __getitem__(self, idx):
        return {
            'seq_tokens': self.seq_tokens[idx],
            'seq_positions': self.seq_positions[idx],
            'target': self.targets[idx]
        }
    
    def __len__(self):
        return len(self.targets)

# 使用标准DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 6.3 训练器集成

```python
# 继承框架的BaseTrainer(如果存在)
# 或参考CTRTrainer实现SeqTrainer

class SeqTrainer:
    def train_one_epoch(self, dataloader):
        for batch in dataloader:
            seq_tokens = batch['seq_tokens'].to(device)
            seq_positions = batch['seq_positions'].to(device)
            target = batch['target'].to(device)
            
            logits = model(seq_tokens, seq_positions)
            last_logits = logits[:, -1, :]
            loss = criterion(last_logits, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 7. 性能优化建议

### 7.1 Gradient Checkpointing

```python
# 在HSTULayer中应用
from torch.utils.checkpoint import checkpoint

def forward(self, X, rel_pos_bias=None):
    # 使用checkpoint减少显存
    X = checkpoint(self._forward_impl, X, rel_pos_bias)
    return X

def _forward_impl(self, X, rel_pos_bias):
    # 实际前向逻辑
    ...
```

### 7.2 Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(seq_tokens, seq_positions)
    loss = criterion(logits, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 7.3 序列长度调整

```python
# 初期使用较短序列
max_seq_len = 128  # 而不是256

# 后期可逐步增加
# max_seq_len = 256
```

---

## 8. 调试和验证清单

- [ ] HSTULayer输入输出维度验证
- [ ] 梯度流动验证(backward pass)
- [ ] 相对位置偏置形状验证
- [ ] 数据加载器输出格式验证
- [ ] 损失函数计算验证
- [ ] 推理输出验证
- [ ] 显存占用监控
- [ ] 训练曲线监控

---

**版本：** v1.0  
**最后更新：** 2025-11-14

