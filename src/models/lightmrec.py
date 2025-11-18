
import os
import random
import math
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss, L2Loss

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=[128, 64], dropout=0.1, activation='relu'):
        super().__init__()
        layers = []
        last_dim = in_dim

        if activation == 'relu':
            act = nn.ReLU()
        elif activation == 'gelu':
            act = nn.GELU()
        elif activation == 'leakyrelu':
            act = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        for h_dim in hidden_dims:
            layers += [
                nn.Linear(last_dim, h_dim),
                nn.BatchNorm1d(h_dim),   
                act,
                nn.Dropout(dropout)
            ]
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    

class PositionalEncoding(nn.Module):
    def __init__(self, pos_dim, max_len, device):
        """
        初始化位置编码。
        :param d_model: 嵌入的维度
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.pos_dim = pos_dim
        self.max_len = max_len
        self.device = device
        self.pe = torch.zeros(max_len, pos_dim, device=self.device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pos_dim, 2).float() * (-math.log(10000.0) / pos_dim))

        self.pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        self.pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        # self.pe = nn.Parameter(torch.zeros(max_len, pos_dim, device=self.device))
        # nn.init.xavier_uniform_(self.pe)


    def forward(self, x):
        """
        将位置编码添加到输入嵌入中。
        :param x: 输入嵌入，形状为 (seq_len, d_model)
        :return: 添加位置编码后的嵌入
        """
        x = x + self.pe
        return x
    
class LightMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(LightMRec, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.alpha = config['alpha']
        self.dropout = config['dropout']
        self.i_pe = PositionalEncoding(config['embedding_size'], max_len=self.n_items, device=self.device)
        self.u_pe = PositionalEncoding(config['embedding_size'], max_len=self.n_users, device=self.device)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
        self.image_trs = MLP(in_dim=self.v_feat.shape[1], 
                             out_dim=self.embedding_dim, 
                             hidden_dims=[256, 128], dropout=0.1, 
                             activation='relu')
        self.text_trs = MLP(in_dim=self.t_feat.shape[1], 
                             out_dim=self.embedding_dim, 
                             hidden_dims=[256, 128], dropout=0.1, 
                             activation='relu')

    def forward(self):
        user_emb = self.user_embedding.weight
        item_image_emb = self.image_trs(self.image_embedding.weight)
        item_text_emb = self.text_trs(self.text_embedding.weight)
        i_v_pos = self.i_pe(item_image_emb)
        i_t_pos = self.i_pe(item_text_emb)
        u_pos = self.u_pe(user_emb)
        alpha = self.alpha
        item_emb  = alpha * i_t_pos + (1 - alpha) * i_v_pos
        return u_pos,item_emb 

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss

    def ssm_loss(self, users, items, user_idx, item_idx):
        neg_edge_index = torch.randint(0, self.n_items, (user_idx.numel(), 32), device=user_idx.device)
        emb_neg = items[neg_edge_index]
        emb1 = users[user_idx]
        emb2 = items[item_idx]
        # emb1 = self.dropout(emb1)
        emb1 = F.normalize(emb1, dim=-1)
        item_emb = torch.cat([emb2.unsqueeze(1), emb_neg], dim=1)
        item_emb = F.normalize(item_emb, dim=-1)
        y_pred = torch.bmm(item_emb, emb1.unsqueeze(-1)).squeeze(-1)
        pos_logits = torch.exp(y_pred[:, 0] / 0.04)
        neg_logits = torch.exp(y_pred[:, 1:] / 0.04)
        Ng = neg_logits.sum(dim=-1)
        loss = (- torch.log(pos_logits / Ng))
        return loss.mean() 

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        # neg_items = interaction[2]

        ua_embeddings, ia_embeddings = self.forward()

        self.build_item_graph = False

        batch_mf_loss = self.ssm_loss(ua_embeddings, ia_embeddings, users, pos_items)
        return batch_mf_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

