---
title: Training & Evaluation
description: Torch-RecHub training and evaluation
---

# Training & Evaluation

Torch-RecHub provides trainers for ranking, matching, multi-task, and sequence-generation models. All provide `fit` and ONNX export, but their batch formats, `evaluate` arguments, and return values are not identical. Follow the contract for each trainer below.

![Trainer workflow](/img/diagrams/trainer_lifecycle.png)

## Experiment Tracking & Visualization

Install the relevant optional dependencies first:

```bash
python -m pip install "torch-rechub[tracking]"
python -m pip install "torch-rechub[visualization]"
python -m pip install "torch-rechub[onnx]"
```

- Supports **WandB / SwanLab / TensorBoardX** as `model_logger`; you can pass a single instance or a list.
- Auto-logs train/validation metrics and hyperparameters: `train/loss`, `learning_rate`, `val/auc` (CTR/Match), `val/task_i_score` (MTL), `val/accuracy` (Seq).
- Set `model_logger=None` (default) for zero overhead when tracking is not needed.

```python
from torch_rechub.basic.tracking import WandbLogger, TensorBoardXLogger
from torch_rechub.trainers import CTRTrainer

wb = WandbLogger(project="rechub-demo", name="deepfm")
tb = TensorBoardXLogger(log_dir="./runs/deepfm")

trainer = CTRTrainer(model, model_logger=[wb, tb])
trainer.fit(train_dataloader, val_dataloader)
```

## Trainers

> **Save directory**: Trainers do not create `model_path` automatically, so create it before calling `fit`. When early stopping actually fires, the trainer restores the recorded best weights before saving. If training reaches its epoch limit without early stopping, it saves the last epoch, which should not be assumed to be the validation-best checkpoint.

### CTRTrainer

Used for ranking (CTR prediction) models such as DeepFM, Wide&Deep, DCN.

```python
import os

from torch_rechub.trainers import CTRTrainer
from torch_rechub.models.ranking import DeepFM

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2})

trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/deepfm"
)

os.makedirs("saved/deepfm", exist_ok=True)
trainer.fit(train_dataloader, val_dataloader)
auc = trainer.evaluate(trainer.model, test_dataloader)
trainer.export_onnx("deepfm.onnx")
trainer.visualization(save_path="deepfm_architecture.pdf")
```

**Parameters**
- `model`: Ranking model instance.
- `optimizer_fn`: Optimizer function, default `torch.optim.Adam`.
- `optimizer_params`: Optimizer parameters.
- `regularization_params`: Dictionary that may contain `embedding_l1`, `embedding_l2`, `dense_l1`, and `dense_l2`.
- `scheduler_fn`: Learning rate scheduler.
- `scheduler_params`: Scheduler parameters.
- `n_epoch`: Number of training epochs.
- `earlystop_patience`: Patience for early stopping.
- `device`: Training device.
- `gpus`: GPU device IDs; more than one uses `torch.nn.DataParallel`.
- `loss_mode`: Boolean. `True` when the model returns only predictions; `False` when the model returns predictions plus auxiliary loss.
- `model_path`: Path to save the model.

### MatchTrainer

Used for matching/retrieval models such as DSSM, YoutubeDNN, MIND.

```python
import os

from torch_rechub.trainers import MatchTrainer
from torch_rechub.models.matching import DSSM

model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # retained by DSSM, but not applied in its current forward()
    user_params={"dims": [256, 128, 64]},
    item_params={"dims": [256, 128, 64]}
)

trainer = MatchTrainer(
    model=model,
    mode=0,  # 0: point-wise, 1: pair-wise, 2: list-wise
    optimizer_params={"lr": 0.001},
    n_epoch=50,
    device="cuda:0",
    model_path="saved/dssm"
)

os.makedirs("saved/dssm", exist_ok=True)
trainer.fit(train_dataloader)
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

**Parameters**
- `model`: Matching model instance.
- `mode`: Training mode, one of 0 (point-wise), 1 (pair-wise), 2 (list-wise).
- `in_batch_neg`: Use in-batch negatives for two-tower models that expose `user_tower()` and `item_tower()`.
- `in_batch_neg_ratio`: Negatives sampled for each positive pair; `None` uses all available negatives.
- `hard_negative`: When `True`, choose the highest-scoring negatives in the current batch.
- `sampler_seed`: Random seed for in-batch negative sampling.
- `optimizer_fn`: Optimizer function, default `torch.optim.Adam`.
- `optimizer_params`: Optimizer parameters.
- `regularization_params`: Dictionary that may contain `embedding_l1`, `embedding_l2`, `dense_l1`, and `dense_l2`.
- `scheduler_fn`: Learning rate scheduler.
- `scheduler_params`: Scheduler parameters.
- `n_epoch`: Number of training epochs.
- `earlystop_patience`: Patience for early stopping.
- `device`: Training device.
- `gpus`: GPU device IDs; more than one uses `torch.nn.DataParallel`.
- `model_path`: Path to save the model.

### MTLTrainer

Used for multi-task models such as MMoE, PLE, ESMM, SharedBottom.

```python
import os

from torch_rechub.trainers import MTLTrainer
from torch_rechub.models.multi_task import MMOE

model = MMOE(
    features=features,
    task_types=["classification", "classification"],
    n_expert=8,
    expert_params={"dims": [32,16]},
    tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}]
)

trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 0.001},
    adaptive_params={"method": "uwl"},
    n_epoch=50,
    earlystop_taskid=0,
    device="cuda:0",
    model_path="saved/mmoe"
)

os.makedirs("saved/mmoe", exist_ok=True)
trainer.fit(train_dataloader, val_dataloader)
trainer.export_onnx("mmoe.onnx")
```

**Parameters**
- `model`: Multi-task model instance.
- `task_types`: List of task types (`classification`, `regression`).
- `optimizer_fn`: Optimizer function, default `torch.optim.Adam`.
- `optimizer_params`: Optimizer parameters.
- `regularization_params`: Dictionary that may contain `embedding_l1`, `embedding_l2`, `dense_l1`, and `dense_l2`.
- `scheduler_fn`: Learning rate scheduler.
- `scheduler_params`: Scheduler parameters.
- `adaptive_params`: Adaptive loss weighting parameters. `None` uses equal weighting; the example uses `{"method": "uwl"}`.
- `n_epoch`: Number of training epochs.
- `earlystop_taskid`: Task id used for early stopping.
- `earlystop_patience`: Patience for early stopping.
- `device`: Training device.
- `gpus`: GPU device IDs; more than one uses `torch.nn.DataParallel`.
- `model_path`: Path to save the model.

> **Early-stopping direction**: `MTLTrainer` currently always treats the score for `earlystop_taskid` as “higher is better.” The default regression metric is MSE, where lower is better, so do not select a regression task as the current early-stopping task.

### SeqTrainer

Used for next-item sequence models such as HSTU and HLLM. Each batch must be `(seq_tokens, seq_positions, seq_time_diffs, targets)`. Evaluation returns `(average_loss, top1_accuracy)`.

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
    loss_type="cross_entropy",  # alternatively "nce"
)

history = trainer.fit(train_dataloader, val_dataloader)
test_loss, top1_accuracy = trainer.evaluate(test_dataloader)
trainer.export_onnx("hstu.onnx", vocab_size=model.vocab_size)
```

`SeqTrainer` ignores `seq_positions` from the batch; current HSTU/HLLM models derive positions internally. `loss_type="cross_entropy"` ignores token `0` by default, so preprocessing must reserve `0` exclusively for padding.

## Evaluation Metrics

Metrics are available from `torch_rechub.basic.metric`:

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

- `auc_score` and `log_loss` accept per-example labels and scores and return floats. Higher AUC is better; lower log loss is better.
- `gauc_score` computes AUC per user. Every included user group must contain both positive and negative labels; pass `weights={user_id: weight}` for custom weighting.
- `topk_metrics` computes NDCG, MRR, Recall, Hit, and Precision. Both inputs are `{user_id: [item_id, ...]}`, and each recommendation list must contain at least `max(topKs)` items.
- `diversity_score`, `coverage_score`, and `novelty_score` measure within-list difference, catalog coverage, and mean self-information respectively; higher is better.

`topk_metrics` and the three beyond-accuracy metrics currently return dictionaries of **formatted string lists**, for example `{"Recall": ["Recall@10: 0.1234"]}`, not dictionaries of raw floats. `ndcg_score`, `hit_score`, `mrr_score`, `recall_score`, and `precision_score` return their corresponding string lists as well.

## Callbacks

### EarlyStopper

Used for early stopping on a metric where higher is better.

```python
from torch_rechub.basic.callback import EarlyStopper

early_stopper = EarlyStopper(patience=10)

if early_stopper.stop_training(auc, model.state_dict()):
    print(f'validation: best auc: {early_stopper.best_auc}')
    model.load_state_dict(early_stopper.best_weights)
    break
```

**Parameters**
- `patience`: Number of consecutive epochs without improvement before stopping.

`EarlyStopper` currently has no `delta`, `mode="min"`, or custom direction option. It cannot directly monitor a loss or MSE that must be minimized.

## Loss Functions

### RegularizationLoss

Supports L1 and L2 regularization.

```python
from torch_rechub.basic.loss_func import RegularizationLoss

reg_loss_fn = RegularizationLoss(
    embedding_l1=0.0,
    embedding_l2=0.0001,
    dense_l1=0.0,
    dense_l2=0.0001
)
```

### BPRLoss

Pairwise loss for matching models.

```python
from torch_rechub.basic.loss_func import BPRLoss

bpr_loss = BPRLoss()
loss = bpr_loss(pos_score, neg_score)
```

