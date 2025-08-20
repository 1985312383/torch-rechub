"""
Date: create on 12/05/2022, update on 20/05/2022
References:
    paper: (CIKM'2013) Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    url: https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf
    code: https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/dssm.py
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn

Refactor:
    Unify DSSM behaviors into a single class via `training_mode`.
    training_mode in {0: point-wise, 1: pair-wise, 2: list-wise}.
    All outputs are divided by `self.temperature` (default 1.0).
"""

from typing import List, Optional

import torch
import torch.nn.functional as F

from ...basic.layers import MLP, EmbeddingLayer


class DSSM(torch.nn.Module):
    """Deep Structured Semantic Model

    Args:
        user_features (list[Feature Class]): features for user tower.
        item_features (list[Feature Class]): positive item features for item tower.
        neg_item_features (list[Feature Class] or None): negative item features.
            - training_mode=0 (point-wise): can be None or empty.
            - training_mode=1 (pair-wise): provide negative item features aligned with item_features.
            - training_mode=2 (list-wise): recommend a SequenceFeature for negative items (e.g., "neg_items").
        user_params (dict): params of User Tower MLP.
        item_params (dict): params of Item Tower MLP.
        temperature (float): temperature factor for similarity score/logits, default 1.0.
        training_mode (int): {0: point-wise, 1: pair-wise, 2: list-wise}. Default 0.

    Note:
        - Method names (user_tower, item_tower, forward) remain unchanged.
        - Default behavior (training_mode=0) equals original DSSM.
    """

    def __init__(
        self,
        user_features: List,
        item_features: List,
        neg_item_features: Optional[List] = None,
        user_params: Optional[dict] = None,
        item_params: Optional[dict] = None,
        temperature: float = 1.0,
        training_mode: int = 0,
    ):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_features = neg_item_features if neg_item_features is not None else []
        self.temperature = temperature
        self.training_mode = training_mode

        # dims
        self.user_dims = sum([fea.embed_dim for fea in self.user_features])
        self.item_dims = sum([fea.embed_dim for fea in self.item_features])

        # embedding layer includes all used features
        all_features = self.user_features + self.item_features + self.neg_item_features
        self.embedding = EmbeddingLayer(all_features)

        # towers
        user_params = {} if user_params is None else user_params
        item_params = {} if item_params is None else item_params
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)

        # inference embedding mode: "user" or "item"
        self.mode: Optional[str] = None

    def forward(self, x):
        user_embedding = self.user_tower(x)  # [B, D]
        item_embedding = self.item_tower(x)

        # Inference embedding export
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        # Training forward by mode
        if self.training_mode == 0:
            # point-wise: sigmoid(sim/temperature)
            y = (user_embedding * item_embedding).sum(dim=1) / self.temperature
            return torch.sigmoid(y)
        elif self.training_mode == 1:
            # pair-wise: (pos, neg) scores / temperature
            pos_emb, neg_emb = item_embedding
            pos_score = (user_embedding * pos_emb).sum(dim=1) / self.temperature
            neg_score = (user_embedding * neg_emb).sum(dim=1) / self.temperature
            return pos_score, neg_score
        else:
            # list-wise: logits over candidates / temperature
            # user: [B, D] -> [B, 1, D]; items: [B, C, D]
            user_embedding = user_embedding.unsqueeze(1)
            logits = (user_embedding * item_embedding).sum(dim=2) / self.temperature
            return logits

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [B, num_user_features*embed_dim] -> [B, D]
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)
        user_embedding = self.user_mlp(input_user)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)  # 始终 [B, D]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            # for user inference, item tower is not used
            return None

        if self.training_mode == 0:
            # point-wise: single positive item, use item_features as-is
            input_item = self.embedding(x, self.item_features, squeeze_dim=True)
            item_embedding = self.item_mlp(input_item)
            item_embedding = F.normalize(item_embedding, p=2, dim=1)
            return item_embedding  # [B, D]

        elif self.training_mode == 1:
            # pair-wise: use explicit positive & negative features from init
            input_item_pos = self.embedding(x, self.item_features, squeeze_dim=True)
            pos_embedding = F.normalize(self.item_mlp(input_item_pos), p=2, dim=1)
            if self.mode == "item":
                return pos_embedding  # [B, D]
            input_item_neg = self.embedding(x, self.neg_item_features, squeeze_dim=True)
            neg_embedding = F.normalize(self.item_mlp(input_item_neg), p=2, dim=1)
            return pos_embedding, neg_embedding

        else:
            # list-wise: use positive item_features and negative neg_item_features (SequenceFeature list)
            pos_emb = self.embedding(x, self.item_features, squeeze_dim=False)  # [B, 1, D]
            pos_emb = F.normalize(pos_emb, p=2, dim=2)
            neg_emb = self.embedding(x, self.neg_item_features, squeeze_dim=False)  # [B, n_neg, D] 或 [B, 1, n_neg, D]
            if neg_emb.dim() == 4:
                neg_emb = neg_emb.squeeze(1)
            neg_emb = F.normalize(neg_emb, p=2, dim=2)
            item_embeddings = torch.cat((pos_emb, neg_emb), dim=1)  # [B, 1+n_neg, D]
            if self.mode == "item":
                return pos_emb.squeeze(1)
            return item_embeddings
