---
title: 常见问题解答
description: Torch-RecHub 常见问题及故障排除指南
---

# 常见问题解答

Torch-RecHub 常见问题及故障排除指南。

## 会推出 TensorFlow 版本吗？

暂不考虑。项目当前以 PyTorch 为唯一运行时，重点是提供易于学习和扩展的推荐模型实现。

## 为什么示例的 AUC 很低或波动很大？

`examples/` 内置的样本数据集只用于验证数据格式、特征定义和训练链路，样本量很小，不用于衡量模型效果。请使用 README 中给出的完整数据集，重新做训练/验证/测试切分后再进行模型对比。

## Annoy 在 Windows 上安装失败怎么办？

先安装项目声明的 Annoy extra：

```bash
python -m pip install "torch-rechub[annoy]"
```

如果 pip 没有找到适合当前 Python/平台的 wheel，就会尝试从源码编译。Windows 上出现 `Microsoft Visual C++ 14.0 or greater is required` 时，请安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，重新打开终端后再执行上述命令。不建议安装与当前 Python 版本不匹配的旧 wheel。

![Annoy 编译环境报错](/img/win_install_annoy_error.png "Annoy 编译环境报错")

## 为什么只安装一个向量后端后，`torch_rechub.serving` 仍然导入失败？

当前 `torch_rechub.serving` 会同时导入 Annoy、Faiss 和 Milvus 实现。如果使用统一的 `builder_factory`，请同时安装三个 extra：

```bash
python -m pip install "torch-rechub[annoy,faiss,milvus]"
```

## 为什么 `fit()` 时提示模型保存路径不存在？

当前 Trainer 不会自动创建 `model_path`，请在训练前创建：

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("saved/deepfm", exist_ok=True)
trainer = CTRTrainer(model, model_path="saved/deepfm")
trainer.fit(train_dataloader, val_dataloader)
```

## 为什么从不同目录运行示例会提示数据文件不存在？

部分历史示例的默认数据路径是相对于脚本所在目录的。请先进入对应的 `examples/ranking`、`examples/matching` 等目录，或通过支持该参数的脚本的 `--dataset_path` 传入明确路径。

## 同一批 Feature 可以同时传给多个模型吗？

不建议。`SparseFeature` 和 `SequenceFeature` 会缓存已创建的 Embedding，复用同一个 Feature 实例会让多个模型共享参数。详见 [Feature 实例与 Embedding 所有权](/zh/core/features#feature-实例与-embedding-所有权)。

