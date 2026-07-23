---
title: Matching Model Training Guide
description: A complete guide to loss functions, similarity metrics, and temperature scaling for matching models
---

# Matching Model Training Guide

## I. So Many Losses? — Three Training Approaches

Matching models generally use one of three training approaches: point-wise, pair-wise, or list-wise. In RecHub, the ***mode*** parameter selects the training approach, and each approach corresponds to a different loss.

### 1.1 Point-wise (mode = 0)

> 🥰**Core idea: Treat matching as binary classification.**

For a matching model, the input is a tuple \<User, Item>, and the output is $P(User, Item)$, representing the user's degree of interest in the item.

The training objective is to make the output as close to 1 as possible for a positive item and as close to 0 as possible for a negative item.

The most common loss is BCELoss (Binary Cross Entropy Loss).

### 1.2 Pair-wise (mode = 1)

> 😝**Core idea: A user's interest in a positive sample should be greater than their interest in a negative sample.**

For a matching model, the input is a triple \<User, ItemPositive, ItemNegative\>. It outputs the interest scores $P(User, ItemPositive)$ and $P(User, ItemNegative)$, representing the user's interest in the positive and negative items.

The training objective is to make the positive sample's interest score as much greater than the negative sample's score as possible.

The framework uses BPRLoss (Bayes Personalized Ranking Loss). The formula is shown below; see [this article](https://www.cnblogs.com/pinard/p/9128682.html "here") for details. The linked formulation differs slightly from the formula below, but the underlying idea is the same.

$$
Loss=\frac{1}{N}\sum^N\ _{i=1}-log(sigmoid(pos\_score - neg\_score))
$$

***

### 1.3 List-wise (mode = 2)

> 😇**Core idea: A user's interest in a positive sample should be greater than their interest in negative samples.**

Isn't that the same as Pair-wise?

Yes! List-wise training follows the same idea as Pair-wise training, but the implementation differs.

For a matching model, the input is an N+2 tuple $<User, ItemPositive, ItemNeg\_1, ... , ItemNeg\_N>$. It outputs the user's interest scores for one positive sample and N negative samples.

The training objective is to make the positive sample's interest score as much greater as possible than the scores of all negative samples.

The framework uses `torch.nn.CrossEntropyLoss`. The model should output unnormalized logits; the loss combines `LogSoftmax` and `NLLLoss` internally, and the label is the position of the positive sample in the candidate list.

> PS: This use of List-wise can easily be confused with listwise learning in ranking. The latter typically optimizes or approximates an objective over an ordered list and is evaluated with order-sensitive metrics such as MAP and NDCG. Here, the task is instead multiclass classification among one positive and several negative samples.

## II. How Far Apart Are Two Vectors? — Three Metrics

> 🤔Given a user vector and an item vector, how should their similarity be measured?

First define a user vector $user \in \mathcal R^D$ and an item vector $item\in \mathcal R^D$, where D is the dimensionality of both vectors.

### 2.1 Cosine

From basic geometry:

$$
cos(a,b)=\frac{<a,b>}{|a|*|b|}
$$

This represents the angle between two vectors and returns a real number in \[-1, 1]. It can therefore serve as a similarity measure: the smaller the angle between the vectors, the more similar they are.

RecHub implementations such as DSSM, YouTubeDNN, MIND, ComiRec, and GRU4Rec L2-normalize the tower outputs and then use their inner product as cosine similarity. Other sequential matching implementations do not all follow exactly the same output convention. When integrating a custom model or switching architectures, consult the corresponding `forward()` method.

### 2.2 Dot Product

This is the inner product of two vectors, written as $<a,b>$ for vectors a and b.

A simple observation is that **if a and b are L2-normalized, i.e. $\tilde{a}=\frac{a}{|a|}\ ,\tilde{b}=\frac{b}{|b|}$, directly taking the inner product of $\tilde{a}$ and $\tilde{b}$ is equivalent to $cos(a,b)$**. (The proof is straightforward and omitted here.)

Several current two-tower and multi-interest implementations use this pattern: compute user/item embeddings, L2-normalize them separately, and then take their inner product. This avoids repeatedly computing vector norms explicitly and makes retrieval with angular/IP indexes convenient, but it is not a universal guarantee across every matching model in the repository.

### 2.3 Euclidean Distance

Euclidean distance is the ordinary meaning of “distance” in everyday life.

> 🙋**For L2-normalized vectors a and b, maximizing cosine similarity is equivalent to minimizing Euclidean distance.**

Why? Consider the following formula:

$$
\begin{align*}
  EuclidianDistance(a,b)^2 &= \sum_{i=1}^N(a_i-b_i)^2 \\
    &= \sum_{i=1}^Na_i^2+\sum_{i=1}^Nb_i^2-\sum_{i=1}^N2*a_i*b_i\\
    &= 2-2*\sum_{i=1}^Na_i*b_i \\
    &= 2*(1-cos(a,b))
\end{align*}
$$

Two details explain the derivation:

1. From the second line to the third, $\sum\ _{i=1}^N a\_i^2=1$ because a is L2-normalized. The same applies to b.
2. From the third line to the fourth, $\sum_{i=1}^Na_i*b_i$ is the inner product of a and b. Because both are L2-normalized, it is equivalent to cosine similarity.

The legacy matching evaluation utility `torch_rechub.utils.match.Annoy` uses the `angular` metric by default, not `euclidean`. For L2-normalized vectors, angular distance is monotonically related to cosine similarity. If you explicitly switch the backend or metric, keep the training score and retrieval metric consistent.

> 🙋**Summary: For two L2-normalized vectors, max dot is equivalent to max cosine, which is equivalent to min EuclideanDistance.**

## III. How Hot Is the Temperature?

> Before continuing, make sure you understand the operations performed by [torch.nn.CrossEntropyLoss](https://blog.csdn.net/sdutstudent/article/details/116097064 "torch.nn.CrossEntropyLoss") (LogSoftmax + NLLLoss), which is also essential for reading the source code. Here is the [official documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html "official documentation").

Consider a List-wise training scenario with one positive sample, three negative samples, and cosine similarity as the training metric.

Suppose the model perfectly predicts a training example and outputs logits of (1, -1, -1, -1). In principle, the Loss should be 0, or at least very small. With CrossEntropyLoss, however, the Loss is:

$$
-log(exp(1)/(exp(1)+exp(-1)*3))=0.341
$$

If instead the logits are divided by a temperature coefficient $temperature=0.2$, they become (5, -5, -5, -5). CrossEntropyLoss then gives:

$$
-log(exp(5)/(exp(5)+exp(-5)*3))=0.016
$$

This produces a negligibly small Loss.

In other words, **dividing logits by a temperature expands the upper and lower bounds of their elements and brings them back into the sensitive range of the softmax operation**.

L2 normalization is commonly paired with temperature scaling, but whether the temperature is actually applied depends on the model. DSSM-SENet and YouTubeDNN, for example, scale their logits, while the base DSSM currently retains the argument without applying it in `forward()`. Do not assume that temperature participates in training merely because a constructor exposes a `temperature` argument.
