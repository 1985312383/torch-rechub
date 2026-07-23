---
title: 训练与评估
description: Torch-RecHub 模型训练与评估
---

# 训练与评估

Torch-RecHub 提供排序、召回、多任务和序列生成训练器。它们都提供 `fit` 和 ONNX 导出，但不同任务的 batch 格式、`evaluate` 参数和返回值不完全相同，请按下方各训练器的约定使用。

![Trainer 工作流图](/img/diagrams/trainer_lifecycle.png)

## 实验跟踪与可视化

先安装对应的可选依赖：

```bash
python -m pip install "torch-rechub[tracking]"
python -m pip install "torch-rechub[visualization]"
python -m pip install "torch-rechub[onnx]"
```

- 支持 **WandB / SwanLab / TensorBoardX** 作为 `model_logger`，可传入单个实例或列表。
- 自动记录训练/验证指标与超参数：`train/loss`、`learning_rate`、`val/auc`（CTR/Match）、`val/task_i_score`（MTL）、`val/accuracy`（Seq）。
- 不需要记录时传 `model_logger=None`（默认）即可零开销。

```python
from torch_rechub.basic.tracking import WandbLogger, TensorBoardXLogger
from torch_rechub.trainers import CTRTrainer

wb = WandbLogger(project="rechub-demo", name="deepfm")
tb = TensorBoardXLogger(log_dir="./runs/deepfm")

trainer = CTRTrainer(model, model_logger=[wb, tb])
trainer.fit(train_dataloader, val_dataloader)
```

## 训练器

> **保存目录**：当前 Trainer 不会自动创建 `model_path`，调用 `fit` 前需要自行创建。当早停真正触发时，Trainer 会先恢复已记录的最佳权重再保存；如果训练轮数结束前没有触发早停，保存的是最后一轮权重，不应默认其一定是验证集最优权重。

### CTRTrainer

用于训练排序模型（CTR预测模型），如DeepFM、Wide&Deep、DCN等。

```python
import os

from torch_rechub.trainers import CTRTrainer
from torch_rechub.models.ranking import DeepFM

# 创建模型
model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2})

# 创建训练器
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/deepfm"
)

# 训练模型
os.makedirs("saved/deepfm", exist_ok=True)
trainer.fit(train_dataloader, val_dataloader)

# 评估模型
auc = trainer.evaluate(trainer.model, test_dataloader)

# 导出ONNX模型
trainer.export_onnx("deepfm.onnx")

# 可视化模型
trainer.visualization(save_path="deepfm_architecture.pdf")
```

**参数说明：**
- `model`：排序模型实例
- `optimizer_fn`：优化器函数，默认torch.optim.Adam
- `optimizer_params`：优化器参数
- `regularization_params`：正则化参数字典，可包含 `embedding_l1`、`embedding_l2`、`dense_l1`、`dense_l2`
- `scheduler_fn`：学习率调度器函数
- `scheduler_params`：学习率调度器参数
- `n_epoch`：训练轮数
- `earlystop_patience`：早停耐心值
- `device`：训练设备
- `gpus`：多 GPU 设备 ID 列表；长度大于 1 时使用 `torch.nn.DataParallel`
- `loss_mode`：损失模式，布尔值。True表示模型只返回预测值，False表示模型返回预测值和额外损失
- `model_path`：模型保存路径

### MatchTrainer

用于训练召回模型，如DSSM、YoutubeDNN、MIND等。

```python
import os

from torch_rechub.trainers import MatchTrainer
from torch_rechub.models.matching import DSSM

# 创建模型
model = DSSM(user_features=user_features, item_features=item_features, temperature=1.0,  # 当前 forward 尚未使用该参数
             user_params={"dims": [256, 128, 64]}, item_params={"dims": [256, 128, 64]})

# 创建训练器
trainer = MatchTrainer(
    model=model,
    mode=0,  # 0: point-wise, 1: pair-wise, 2: list-wise
    optimizer_params={"lr": 0.001},
    n_epoch=50,
    device="cuda:0",
    model_path="saved/dssm"
)

# 训练模型
os.makedirs("saved/dssm", exist_ok=True)
trainer.fit(train_dataloader)

# 导出用户塔ONNX模型
trainer.export_onnx("user_tower.onnx", mode="user")

# 导出物品塔ONNX模型
trainer.export_onnx("item_tower.onnx", mode="item")
```

**参数说明：**
- `model`：召回模型实例
- `mode`：训练模式，可选值：0（point-wise）、1（pair-wise）、2（list-wise）
- `in_batch_neg`：是否对具有 `user_tower()` / `item_tower()` 的双塔模型使用 batch 内负采样
- `in_batch_neg_ratio`：每个正样本从当前 batch 中采样的负样本数；为 `None` 时使用可用负样本
- `hard_negative`：为 `True` 时选择当前 batch 内得分最高的负样本
- `sampler_seed`：batch 内随机负采样的随机种子
- `optimizer_fn`：优化器函数，默认torch.optim.Adam
- `optimizer_params`：优化器参数
- `regularization_params`：正则化参数字典，可包含 `embedding_l1`、`embedding_l2`、`dense_l1`、`dense_l2`
- `scheduler_fn`：学习率调度器函数
- `scheduler_params`：学习率调度器参数
- `n_epoch`：训练轮数
- `earlystop_patience`：早停耐心值
- `device`：训练设备
- `gpus`：多 GPU 设备 ID 列表；长度大于 1 时使用 `torch.nn.DataParallel`
- `model_path`：模型保存路径

### MTLTrainer

用于训练多任务模型，如MMoE、PLE、ESMM、SharedBottom等。

```python
import os

from torch_rechub.trainers import MTLTrainer
from torch_rechub.models.multi_task import MMOE

# 创建模型
model = MMOE(features=features, task_types=["classification", "classification"], n_expert=8,
             expert_params={"dims": [32,16]}, tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

# 创建训练器
trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 0.001},
    adaptive_params={"method": "uwl"},  # 自适应损失权重方法
    n_epoch=50,
    earlystop_taskid=0,  # 早停依赖的任务ID
    device="cuda:0",
    model_path="saved/mmoe"
)

# 训练模型
os.makedirs("saved/mmoe", exist_ok=True)
trainer.fit(train_dataloader, val_dataloader)

# 导出ONNX模型
trainer.export_onnx("mmoe.onnx")
```

**参数说明：**
- `model`：多任务模型实例
- `task_types`：任务类型列表，可选值：classification、regression
- `optimizer_fn`：优化器函数，默认torch.optim.Adam
- `optimizer_params`：优化器参数
- `regularization_params`：正则化参数字典，可包含 `embedding_l1`、`embedding_l2`、`dense_l1`、`dense_l2`
- `scheduler_fn`：学习率调度器函数
- `scheduler_params`：学习率调度器参数
- `adaptive_params`：自适应损失权重参数；`None` 表示等权平均，本页示例使用 `{"method": "uwl"}`
- `n_epoch`：训练轮数
- `earlystop_taskid`：早停依赖的任务ID
- `earlystop_patience`：早停耐心值
- `device`：训练设备
- `gpus`：多 GPU 设备 ID 列表；长度大于 1 时使用 `torch.nn.DataParallel`
- `model_path`：模型保存路径

> **早停指标方向**：`MTLTrainer` 当前始终把 `earlystop_taskid` 对应的分数当作“越大越好”。回归任务的默认指标是 MSE（越小越好），因此不要将回归任务选为当前早停主任务。

### SeqTrainer

用于 HSTU/HLLM 这类 next-item 序列模型。数据 batch 必须是 `(seq_tokens, seq_positions, seq_time_diffs, targets)`；评估返回 `(平均损失, top-1 准确率)`。

```python
import os

from torch_rechub.trainers import SeqTrainer

os.makedirs("saved/hstu", exist_ok=True)
trainer = SeqTrainer(
    model=model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=10,
    device="cpu",
    model_path="saved/hstu",
    loss_type="cross_entropy",  # 另可选 "nce"
)

history = trainer.fit(train_dataloader, val_dataloader)
test_loss, top1_accuracy = trainer.evaluate(test_dataloader)
trainer.export_onnx("hstu.onnx", vocab_size=model.vocab_size)
```

`SeqTrainer` 不使用传入 batch 的 `seq_positions`；当前 HSTU/HLLM 由模型内部推导位置。`loss_type="cross_entropy"` 默认忽略 token `0`，因此数据处理需要保证 `0` 专用于 padding。

## 评估指标

指标入口位于 `torch_rechub.basic.metric`：

```python
from torch_rechub.basic.metric import (
    auc_score,
    coverage_score,
    diversity_score,
    gauc_score,
    log_loss,
    novelty_score,
    topk_metrics,
)

auc = auc_score(y_true, y_score)
gauc = gauc_score(y_true, y_score, user_ids)
ranking = topk_metrics(ground_truth, recommendations, topKs=[5, 10])
diversity = diversity_score(recommendations, item_embeddings, topKs=[10])
coverage = coverage_score(recommendations, all_item_ids, topKs=[10])
novelty = novelty_score(recommendations, item_popularity, topKs=[10])
```

- `auc_score` 和 `log_loss` 接收逐样本标签/分数并返回浮点数；AUC 越大越好，log loss 越小越好。
- `gauc_score` 按用户分组计算 AUC；每个参与计算的用户分组都需要同时包含正负标签，可通过 `weights={user_id: weight}` 自定义加权。
- `topk_metrics` 计算 NDCG、MRR、Recall、Hit 和 Precision。`ground_truth` 与 `recommendations` 都是 `{user_id: [item_id, ...]}`，推荐列表至少需要包含 `max(topKs)` 个物品。
- `diversity_score`、`coverage_score`、`novelty_score` 分别衡量列表内差异、目录覆盖和平均自信息，都是越大越好。

`topk_metrics` 和三个超越准确率指标当前返回的是按指标分组的**已格式化字符串列表**，例如 `{"Recall": ["Recall@10: 0.1234"]}`，不是纯浮点数字典。`ndcg_score`、`hit_score`、`mrr_score`、`recall_score`、`precision_score` 也返回对应的字符串列表。

## 回调函数

### EarlyStopper

用于早停，当验证集上“越大越好”的指标不再提升时停止训练。

```python
from torch_rechub.basic.callback import EarlyStopper

# 创建早停器
early_stopper = EarlyStopper(patience=10)

# 在训练过程中使用
if early_stopper.stop_training(auc, model.state_dict()):
    print(f'validation: best auc: {early_stopper.best_auc}')
    model.load_state_dict(early_stopper.best_weights)
    break
```

**参数说明：**
- `patience`：早停耐心值，即连续多少轮验证集性能没有提升就停止训练

`EarlyStopper` 当前没有 `delta`、`mode="min"` 或自定义指标方向参数，不能直接用来监控需要最小化的 loss/MSE。

## 损失函数

### RegularizationLoss

用于正则化，支持L1和L2正则化。

```python
from torch_rechub.basic.loss_func import RegularizationLoss

# 创建正则化损失函数
reg_loss_fn = RegularizationLoss(
    embedding_l1=0.0,  # Embedding层L1正则化系数
    embedding_l2=0.0001,  # Embedding层L2正则化系数
    dense_l1=0.0,  # 密集层L1正则化系数
    dense_l2=0.0001  # 密集层L2正则化系数
)
```

### BPRLoss

用于召回模型的 pairwise 损失。

```python
from torch_rechub.basic.loss_func import BPRLoss

# 创建BPR损失函数
bpr_loss = BPRLoss()

# 计算损失
loss = bpr_loss(pos_score, neg_score)
```
