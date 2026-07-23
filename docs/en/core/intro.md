---
title: Core Components Overview
description: Overview of Torch-RecHub's core components
---

# Core Components Overview

Torch-RecHub uses a modular design that divides the core functionality of a recommender system into components for feature processing, data pipelines, model design, training, and evaluation. This modular design makes the framework easy to use and extend while preserving a well-organized codebase.

## Core Component Architecture

Torch-RecHub's core component architecture is organized as follows:

1. **Feature layer**: Handles different types of features, including numerical, categorical, and sequence features
2. **Data layer**: Handles data loading, preprocessing, and DataLoader generation
3. **Model layer**: Implements recommendation models, including ranking, matching, multi-task, and generative recommendation models
4. **Training layer**: Provides unified training interfaces for model training, evaluation, prediction, and ONNX export
5. **Tools layer**: Provides utilities such as ONNX export, model visualization, callbacks, and loss functions

## Component Relationships

The core components relate to one another as follows:

1. **Feature layer** -> **Data layer**: Feature definitions guide preprocessing and feature engineering in the data layer
2. **Data layer** -> **Training layer**: DataLoaders produced by data generators are used for model training and evaluation
3. **Model layer** -> **Training layer**: The training layer trains and evaluates models defined in the model layer
4. **Training layer** -> **Tools layer**: The training layer uses utilities for tasks such as ONNX export and model visualization

## Component Details

### Feature Processing

The feature-processing component defines and handles different types of features, including:

- **DenseFeature**: Handles numerical features
- **SparseFeature**: Handles categorical features
- **SequenceFeature**: Handles sequence or multi-hot features

See [Feature Definitions](/core/features) for details.

### Data Pipeline

The data-pipeline component handles data loading, preprocessing, and DataLoader generation, including:

- **TorchDataset**: A dataset for training and validation
- **PredictDataset**: A dataset for prediction
- **DataGenerator**: Generates DataLoaders for ranking and multi-task models
- **MatchDataGenerator**: Generates DataLoaders for matching models
- **SequenceDataGenerator**: Generates DataLoaders for HSTU/HLLM sequence tasks
- **ParquetIterableDataset**: Streams Parquet files batch by batch

See [Data Pipeline](/core/data) for details.

### Training and Evaluation

The training-and-evaluation component trains different types of recommendation models, including:

- **CTRTrainer**: Trains ranking models
- **MatchTrainer**: Trains matching models
- **MTLTrainer**: Trains multi-task models
- **SeqTrainer**: Trains HSTU/HLLM sequence-generation models

RQ-VAE and TIGER use dedicated data-processing and training workflows. See [Generative Models](/models/generative) and the corresponding reproduction guides.

See [Training and Evaluation](/core/evaluation) for details.
