---
title: 召回模型训练指南
description: 召回模型的损失函数、相似度度量和温度缩放的完整指南
---

# 召回模型训练指南

## 一、乱七八糟的Loss？—— 3种训练方式

召回中，一般的训练方式分为三种：point-wise、pair-wise、list-wise。在RecHub中，用参数***mode***来指定训练方式，每一种不同的训练方式也对应不同的Loss。

### 1.1 Point wise (mode = 0)

> 🥰**核心思想：将召回视为二分类。**

对于一个召回模型，输入二元组\<User, Item>，输出$P(User, Item)$，表示User对Item的感兴趣程度。

训练目标为：若物品为正样本，输出应尽可能接近1，负样本则输出尽可能接近0。

采用的Loss最常见的就是BCELoss（Binary Cross Entropy Loss）。

### 1.2 Pair wise (mode = 1)

> 😝**核心思想：用户对正样本感兴趣的程度应该大于负样本。**

对于一个召回模型，输入三元组\<User, ItemPositive, ItemNegative\>，输出兴趣得分$P(User, ItemPositive)$，$P(User, ItemNegative)$，表示用户对正样本物品和负样本物品的兴趣得分。

训练目标为：正样本的兴趣得分应尽可能大于负样本的兴趣得分。

框架中采用的Loss为BPRLoss（Bayes Personalized Ranking Loss）。Loss的公式这里放一个公式，详细可以参考[这里](https://www.cnblogs.com/pinard/p/9128682.html "这里")（链接里的内容和下面的公式有些细微的差别，但是思想是一样的）

$$
Loss=\frac{1}{N}\sum^N\ _{i=1}-log(sigmoid(pos\_score - neg\_score))
$$

***

### 1.3 List wise（mode = 2）

> 😇**核心思想：用户对正样本感兴趣的程度应该大于负样本 。**

嗯？怎么和Pair wise一样？

没错！List wise的训练方式的思想和Pair wise是一样的，只不过实现上不同。

对于一个召回模型，输入N+2元组$<User, ItemPositive, ItemNeg\_1, ... , ItemNeg\_N>$，输出用户对1个正样本和N个负样本的兴趣得分。

训练目标为：对正样本的兴趣得分应该尽可能大于其他所有负样本的兴趣得分。

框架中采用的 Loss 为 `torch.nn.CrossEntropyLoss`。模型应输出未归一化 logits；该损失内部组合 `LogSoftmax` 与 `NLLLoss`，标签是正样本在候选列表中的位置。

> PS：这里的 List wise 容易和 Ranking 中的 listwise 混淆。后者通常直接优化或近似排序列表目标，并用 MAP、NDCG 等顺序敏感指标评价；这里则是在一个正样本与若干负样本之间做多分类。

## 二、两个向量有多远？—— 3 种衡量指标

> 🤔给定一个用户向量和一个物品向量，如何衡量他们之间的相似度？

先定义用户向量$user \in \mathcal R^D$，$item\in \mathcal R^D$，D表示用户向量和物品向量的维度。

### 2.1 cosine

初中学过：

$$
cos(a,b)=\frac{<a,b>}{|a|*|b|}
$$

表示两个向量的夹角，会输出一个\[-1, 1]之间的实数，我们就可以以此作为相似度的衡量依据：两个向量之间角度越小，就越相似。

RecHub 中 DSSM、YouTubeDNN、MIND、ComiRec、GRU4Rec 等实现会对塔输出做 L2 归一化，再以内积得到 cosine 相似度；其他序列召回实现的输出约定并不完全相同。接入自定义模型或切换架构时，应以对应 `forward()` 为准。

### 2.2 dot

即向量的内积，用$<a,b> $表示向量a、b的内积。

一个很简单的思想是：**如果将a、b 向量L2 normalize，即 $\tilde{a}=\frac{a}{|a|}\ ,\tilde{b}=\frac{b}{|b|}$，然后直接将 $\tilde{a}、\tilde{b}$求内积，就等价于于 $cos(a,b)$**。 （很容易，这里就不推导了）

当前多种双塔/多兴趣实现采用这种写法：先计算 user/item embedding，分别做 L2 normalize，再计算内积。它避免重复显式计算向量范数，也便于用 angular/IP 索引检索；但这不是仓库内所有 matching 模型的统一保证。

### 2.3 Euclidian Distance（欧氏距离）

欧氏距离即我们生活中“距离”的含义。

> 🙋**经过L2 Norm后的向量a,b，最大化其cosine相似度与最小化其欧氏距离，是等价的**

为什么？见下面公式：

$$
\begin{align*}
  EuclidianDistance(a,b)^2 &= \sum_{i=1}^N(a_i-b_i)^2 \\
    &= \sum_{i=1}^Na_i^2+\sum_{i=1}^Nb_i^2-\sum_{i=1}^N2*a_i*b_i\\
    &= 2-2*\sum_{i=1}^Na_i*b_i \\
    &= 2*(1-cos(a,b))
\end{align*}
$$

两点解释一下：

1. 第二行到第三行，$\sum\ _{i=1}^N a\_i^2=1$。为什么？因为a是L2 Norm后的向量。b同理。
2. 第三行到第四行，$\sum_{i=1}^Na_i*b_i$，即向量a、b的内积，因为a、b已经L2 Norm，所以相当于cos。

旧版 matching 评估工具 `torch_rechub.utils.match.Annoy` 默认使用 `angular` 度量，而不是 `euclidean`。对 L2 归一化向量，angular 距离与 cosine 相似度单调对应；若显式切换后端或 metric，训练打分和检索度量应保持一致。

> 🙋**小结：L2 Norm后的两个向量，max dot等价于max cosine等价于min EuclidianDistance**

## 三、 温度系数有多热？

> 在此之前，请先确认明白[torch.nn.CrossEntropyLoss](https://blog.csdn.net/sdutstudent/article/details/116097064 "torch.nn.CrossEntropyLoss")中都进行了什么运算（LogSoftmax + NLLLoss），这对阅读源码也是关键的信息。这里是[官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html "官方文档")。

假设一个场景：采用List wise的训练方式，1个正样本，3个负样本，cosine相似度作为训练过程中的衡量指标。

假设此时我的模型完美的预测了一条训练数据，即输出的logits为（1, -1, -1, -1），按理说我的Loss应该为0，至少应该很小。但此时如果采用CrossEntropyLoss，得到的Loss是：

$$
-log(exp(1)/(exp(1)+exp(-1)*3))=0.341
$$

但此时如果对logits除上一个温度系数$temperature=0.2 $，即logits为（5, -5, -5, -5），经过CrossEntropyLoss，得到的Loss是：

$$
-log(exp(5)/(exp(5)+exp(-5)*3))=0.016
$$

这样就会得到一个很小到可以忽略不计的Loss了。

也就是说，**对logits除上一个temperature的作用是扩大logits中每个元素中的上下限，拉回softmax运算的敏感范围** 。

L2 normalize 常与 temperature 搭配使用，但 temperature 是否生效取决于具体模型：例如 DSSM-SENet、YouTubeDNN 会缩放 logits，而基础 DSSM 当前保留了参数但未在 `forward()` 中应用。不要仅凭构造函数存在 `temperature` 参数就假设它已参与训练。
