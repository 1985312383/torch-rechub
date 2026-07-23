---
title: Ranking Model Tutorial
description: Torch-RecHub ranking model tutorial covering WideDeep, DeepFM, DCN, and entry points for sequence ranking models
---

# Ranking Model Tutorial

This tutorial focuses on the shared training workflow for ranking scenarios: data preparation, feature definition, trainer usage, evaluation, and common extensions. The basic examples use the built-in `Criteo` sample data, while the sequence-model section uses the `Amazon Electronics` sample data.

## I. Basic Ranking Pipeline (Criteo)

### 1. Data Preparation and Feature Processing

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

df = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

# Ranking baselines generally follow this "continuous features + categorical features" input pattern.
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# Keep missing-value handling aligned with the repository examples to avoid reproduction differences.
df[sparse_features] = df[sparse_features].fillna("-996")
df[dense_features] = df[dense_features].fillna(0)

# Normalize continuous features and encode categorical features as discrete IDs.
scaler = MinMaxScaler()
df[dense_features] = scaler.fit_transform(df[dense_features])

for feat in sparse_features:
    encoder = LabelEncoder()
    df[feat] = encoder.fit_transform(df[feat].astype(str))

# These Feature objects describe how each column should be fed into the model.
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=df[name].nunique(), embed_dim=16) for name in sparse_features]

x = df.drop(columns=["label"])
y = df["label"]

# DataGenerator automatically splits the training, validation, and test sets according to split_ratio.
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)
```

### 2. Shared Training Pattern for WideDeep / DeepFM / DCN

```python
import os
from torch_rechub.models.ranking import WideDeep, DeepFM, DCN
from torch_rechub.trainers import CTRTrainer

# Choose any one of the models.
# DeepFM is a good first ranking pipeline; the commented WideDeep / DCN variants only switch the model and do not change the data flow.
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

# model = WideDeep(
#     wide_features=sparse_feas,
#     deep_features=sparse_feas + dense_feas,
#     mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
# )

# model = DCN(
#     features=dense_feas + sparse_feas,
#     n_cross_layers=3,
#     mlp_params={"dims": [256, 128]},
# )

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=2,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path="./saved/ctr_basic",
)

# Create the save directory before training so saving the best weights after fit does not fail.
os.makedirs("./saved/ctr_basic", exist_ok=True)
trainer.fit(train_dl, val_dl)
# Passing trainer.model to evaluate evaluates the best model currently held by the trainer.
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 3. Which Models Does This Section Apply To?

- `WideDeep`: Quickly validate a wide + deep architecture
- `DeepFM`: A classic baseline combining categorical-feature interactions with an MLP
- `DCN / DCNv2`: Explicit feature crossing

These models all use the same `DenseFeature + SparseFeature + DataGenerator + CTRTrainer` training pattern.

## II. Sequence Ranking Pipeline (DIN / BST)

The main difference between sequence models and basic ranking models is that you must generate historical behavior sequences and keep `history_features` strictly aligned with `target_features`.

### 1. Build Sequences from the Amazon Electronics Sample Data

```python
import pandas as pd

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# generate_seq_feature sorts by time and generates historical item / category sequences for each sample.
train, val, test = generate_seq_feature(
    data=data,
    user_col="user_id",
    item_col="item_id",
    time_col="time",
    item_attribute_cols=["cate_id"],
)

n_users = data["user_id"].max()
n_items = data["item_id"].max()
n_cates = data["cate_id"].max()

# features contains only user-profile/context features; target_features and history_features correspond one to one.
features = [
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8),
]
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
]

history_features = [
    # Sequence models must retain the complete sequence tensor here, so use concat rather than mean / sum.
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat", shared_with="target_item_id"),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat", shared_with="target_cate_id"),
]

# Convert each DataFrame to a dict before passing it to DataGenerator, matching the examples/ranking workflow.
train_dict, val_dict, test_dict = df_to_dict(train), df_to_dict(val), df_to_dict(test)
train_y = train_dict.pop("label")
val_y = val_dict.pop("label")
test_y = test_dict.pop("label")

dg = DataGenerator(train_dict, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val_dict,
    y_val=val_y,
    x_test=test_dict,
    y_test=test_y,
    batch_size=4096,
)
```

### 2. Creating DIN / BST

```python
import os
from torch_rechub.models.ranking import DIN, BST
from torch_rechub.trainers import CTRTrainer

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    attention_mlp_params={"dims": [256, 128]},
)

# model = BST(
#     features=features,
#     history_features=history_features,
#     target_features=target_features,
#     mlp_params={"dims": [256, 128]},
#     nhead=8,
#     dropout=0.2,
#     num_layers=1,
# )

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=2,
    earlystop_patience=2,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path="./saved/ctr_sequence",
)

# Again, create the save directory first so saving weights after training does not fail.
os.makedirs("./saved/ctr_sequence", exist_ok=True)
trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 3. Key Constraints for Sequence Models

- `SequenceFeature` must use `pooling="concat"` because DIN / DIEN / BST need the complete sequence tensor.
- `history_features` and `target_features` must correspond one to one and share embeddings through `shared_with`.
- `features` contains user-profile/context features. Do not put all of `target_features` into it again, or the inputs will be duplicated; BST will also fail because the history and target dimensions differ.
- For `BST`, `embed_dim` must be divisible by `nhead`.

DIEN also requires a per-timestep negative-sample sequence in `neg_history_features`, and `CTRTrainer` must set `loss_mode=False`; you cannot simply replace the model class above. Refer directly to the [DIEN Tutorial](/tutorials/models/ranking/dien).

## III. Evaluation, Export, and Visualization

### 1. Evaluation

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 2. ONNX Export

```python
trainer.export_onnx("model.onnx", dynamic_batch=True)
```

### 3. Architecture Visualization

```python
from torch_rechub.utils.visualization import visualize_model

visualize_model(model, save_path="model_architecture.png", dpi=300)
```

> Visualization requires an additional dependency: `pip install "torch-rechub[visualization]"`

## IV. FAQ

### Q1: Why Doesn't This Page Include Complete Code for Every Ranking Model?

Ranking models can be divided into two groups:

- Basic ranking: `WideDeep / DeepFM / DCN`
- Sequence ranking: `DIN / DIEN / BST`

Their data-preparation workflows differ. This page preserves the shared workflow, while each model page covers model-specific parameters and tuning guidance.

### Q2: How Do I Switch to a GPU?

Change `device="cpu"` to `device="cuda:0"`.

### Q3: Why Does the Example Call `os.makedirs` First?

The current `CTRTrainer` saves weights directly to `model_path` and does not create the directory automatically. Create the save directory before training so the example can run as written.

### Q4: Where Can I Find More Detailed Model Guides?

- [DeepFM Tutorial](/tutorials/models/ranking/deepfm)
- [WideDeep Tutorial](/tutorials/models/ranking/widedeep)
- [DCN Tutorial](/tutorials/models/ranking/dcn)
- [DIN Tutorial](/tutorials/models/ranking/din)
- [DIEN Tutorial](/tutorials/models/ranking/dien)
- [BST Tutorial](/tutorials/models/ranking/bst)
