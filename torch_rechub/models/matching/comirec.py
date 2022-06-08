"""
Date: create on 07/06/2022
References: 
    paper: Controllable Multi-Interest Framework for Recommendation
    url: https://arxiv.org/pdf/2005.09347.pdf
Authors: Kai Wang, 306178200@qq.com
"""

import torch

from ...basic.layers import MLP, EmbeddingLayer, MultiInterestSA, CapsuleNetwork
from torch import nn


class ComirecSA(torch.nn.Module):
    """The match model mentioned in `Controllable Multi-Interest Framework for Recommendation` paper.
    It's a Comirec match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        history_features (list[Feature Class]): training history
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        sim_func (str): similarity function, includes `["cosine", "dot"]`, default to "cosine".
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, history_features, item_features, neg_item_feature, user_params, sim_func="cosine", temperature=1.0, interest_num=4):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_feature = neg_item_feature
        self.sim_func = sim_func
        self.temperature = temperature
        self.interest_num = interest_num
        self.user_dims = sum([fea.embed_dim for fea in user_features+history_features])

        self.embedding = EmbeddingLayer(user_features + item_features + history_features)
        self.multi_interest_sa = MultiInterestSA(embedding_dim=self.history_features[0].embed_dim, interest_num=self.interest_num)
        self.convert_user_weight = nn.Parameter(torch.rand(self.user_dims, self.history_features[0].embed_dim), requires_grad=True)
        # self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        pos_item_embedding = item_embedding[:,0,:]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)
        best_interest_emb = torch.rand(user_embedding.shape[0], user_embedding.shape[2]).to(user_embedding.device)
        for k in range(user_embedding.shape[0]):
            best_interest_emb[k, :] = user_embedding[k, k_index[k], :]
        best_interest_emb = best_interest_emb.unsqueeze(1)

        if self.sim_func == "cosine":
            y = torch.cosine_similarity(best_interest_emb, item_embedding, dim=-1)  #[batch_size, 1+n_neg_items, embed_dim]
        elif self.sim_func == "dot":
            y = torch.mul(best_interest_emb, item_embedding).sum(dim=1)
        else:
            raise ValueError("similarity function only support %s, but got %s" % (["cosine", "dot"], self.sim_func))
        y = y / self.temperature
        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True).unsqueeze(1)  #[batch_size, num_features*deep_dims]
        input_user = input_user.expand([input_user.shape[0], self.interest_num, input_user.shape[-1]])

        history_emb = self.embedding(x, self.history_features).squeeze(1)
        mask = self.gen_mask(x)
        mask = mask.unsqueeze(-1).float()
        multi_interest_emb = self.multi_interest_sa(history_emb,mask)

        input_user = torch.cat([input_user,multi_interest_emb],dim=-1)

        # user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, interest_num, embed_dim]
        user_embedding = torch.matmul(input_user,self.convert_user_weight)
        if self.mode == "user":
            return user_embedding  #inference embedding mode -> [batch_size, interest_num, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.squeeze(1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False).squeeze(1)  #[batch_size, n_neg_items, embed_dim]
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]

    def gen_mask(self, x):
        his_list = x[self.history_features[0].name]
        mask = (his_list > 0).long()
        return mask