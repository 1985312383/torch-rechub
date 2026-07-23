---
title: Benchmark (Experimental)
description: Reproducible model comparison benchmarks for Torch-RecHub
---

# Benchmark (Experimental)

`benchmarks/` provides reproducible model comparisons under a fixed experimental protocol and is maintained separately from `examples/`: `examples/` focuses on teaching, while `benchmarks/` focuses on side-by-side comparisons.

Install the YAML extra before running benchmarks from an installed package. The same command can also be used to add the dependency in a repository development environment:

```bash
pip install "torch-rechub[benchmark]"
```

## Quick Start

Run these commands from the repository root:

```bash
# Single config
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml

# Run a group and produce a comparison table
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching
```

Each run writes:

- `result.yaml`: structured metrics and run metadata (git commit, Python/Torch versions, and timestamp)
- `summary.md`: a human-readable summary
- Model weights: `model.pth` for Matching/Ranking; currently `model_base_0.pth` for Multi-Task

A suite additionally writes `suite.yaml` and `suite.md` comparison tables, for example:

| model | dataset | params | Hit@10 | NDCG@10 | train_s |
| --- | --- | --- | --- | --- | --- |
| MIND | ml-1m-sample | 3792 | 0.5000 | 0.5000 | 0.026 |
| YoutubeDNN | ml-1m-sample | 23872 | 0.0000 | 0.0000 | 0.021 |
| ComirecDR | ml-1m-sample | 54736 | 0.0000 | 0.0000 | 0.023 |
| ComirecSA | ml-1m-sample | 4816 | 0.0000 | 0.0000 | 0.018 |

## Supported Tasks

### Matching / Retrieval

Dataset: MovieLens sample (`examples/matching/data/ml-1m/ml-1m_sample.csv`)

| Config | Model |
| --- | --- |
| `configs/matching/ml_1m_mind.yaml` | MIND |
| `configs/matching/ml_1m_youtube_dnn.yaml` | YoutubeDNN |
| `configs/matching/ml_1m_comirec_dr.yaml` | ComirecDR |
| `configs/matching/ml_1m_comirec_sa.yaml` | ComirecSA |

Metrics: `Hit@K`, `Recall@K`, `NDCG@K`, `MRR@K`, and `Precision@K`

### Ranking / CTR

Dataset: Criteo sample (`examples/ranking/data/criteo/criteo_sample.csv`)

| Config | Model |
| --- | --- |
| `configs/ranking/criteo_widedeep.yaml` | WideDeep |
| `configs/ranking/criteo_deepfm.yaml` | DeepFM |
| `configs/ranking/criteo_dcn.yaml` | DCN |

Metrics: `AUC` and `LogLoss` (customizable through the YAML `metrics:` field; supported values are `AUC`, `LogLoss`, `Accuracy`, and `MSE`)

### Multi-Task

Dataset: Census-Income sample (`examples/ranking/data/census-income/`), with two tasks: income (`cvr`) and marital status (`ctr`)

| Config | Model |
| --- | --- |
| `configs/multitask/census_esmm.yaml` | ESMM |
| `configs/multitask/census_mmoe.yaml` | MMOE |
| `configs/multitask/census_ple.yaml` | PLE |

Metrics: `AUC[<task>]` for each task plus `AUC_mean` (tasks whose value is NaN are automatically excluded)

## Config Format

```yaml
task: matching          # matching | ranking | multitask

dataset:
  name: ml-1m-sample
  path: examples/matching/data/ml-1m/ml-1m_sample.csv
  seq_max_len: 50
  neg_ratio: 3

model:
  name: MIND
  params:
    embed_dim: 16
    interest_num: 4
    temperature: 0.02

trainer:
  mode: 2
  epochs: 1
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.000001
  device: cpu
  seed: 2022

metrics:
  topk: 10

output_dir: benchmark_results/matching/ml_1m_mind
```

The loader validates the schema. A misspelled key such as `batchsize` raises an error instead of silently falling back to a default value.

## Baselines and Regression Checks

`benchmarks/baselines/<task>.yaml` records the `expected + tolerance` values for each config. With `--check-baseline`, the process exits non-zero when a metric misses its expectation by more than the allowed tolerance:

```bash
python benchmarks/runner.py \
  --config benchmarks/configs/ranking/criteo_dcn.yaml \
  --check-baseline

python benchmarks/suite.py \
  --configs "benchmarks/configs/ranking/*.yaml" \
  --output  benchmark_results/suites/ranking \
  --check-baseline
```

**Updating baselines**: run a suite, copy the metrics from `benchmark_results/suites/<task>/suite.yaml` to `benchmarks/baselines/<task>.yaml`, and set an appropriate `tolerance`.

Regression direction is metric-aware: higher-is-better metrics such as `AUC`, `Hit`, and `Recall` fail only below `expected - tolerance`; lower-is-better metrics such as `LogLoss` and `MSE` fail only above `expected + tolerance`.
