# coding: utf-8
# @email: enoche.chow@gmail.com
r"""
FREEDOM: A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation
# Update: 01/08/2022
"""


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
from utils.utils import build_sim, compute_normalized_laplacian
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
class FREEDOM(GeneralRecommender):
    def __init__(self, config, dataset):
        super(FREEDOM, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.knn_k = config['knn_k']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.build_item_graph = True
        self.mm_image_weight = config['mm_image_weight']
        self.dropout = config['dropout']
        self.i_pe = PositionalEncoding(config['embedding_size'], max_len=self.n_items, device=self.device)
        self.u_pe = PositionalEncoding(config['embedding_size'], max_len=self.n_users, device=self.device)
        self.degree_ratio = config['degree_ratio']

        self.n_nodes = self.n_users + self.n_items

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.masked_adj, self.mm_adj = None, None
        self.edge_indices, self.edge_values = self.get_edge_info()
        self.edge_indices, self.edge_values = self.edge_indices.to(self.device), self.edge_values.to(self.device)
        self.edge_full_indices = torch.arange(self.edge_values.size(0)).to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        mm_adj_file = os.path.join(dataset_path, 'mm_adj_freedomdsp_{}_{}.pt'.format(self.knn_k, int(10*self.mm_image_weight)))

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = MLP(in_dim=self.v_feat.shape[1], 
                             out_dim=self.embedding_dim, 
                             hidden_dims=[256, 128], dropout=0.1, 
                             activation='relu')
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = MLP(in_dim=self.t_feat.shape[1], 
                             out_dim=self.embedding_dim, 
                             hidden_dims=[256, 128], dropout=0.1, 
                             activation='relu')

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        # construct sparse adj
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def pre_epoch_processing(self):
        if self.dropout <= .0:
            self.masked_adj = self.norm_adj
            return
        # degree-sensitive edge pruning
        degree_len = int(self.edge_values.size(0) * (1. - self.dropout))
        degree_idx = torch.multinomial(self.edge_values, degree_len)
        # random sample
        keep_indices = self.edge_indices[:, degree_idx]
        # norm values
        keep_values = self._normalize_adj_m(keep_indices, torch.Size((self.n_users, self.n_items)))
        all_values = torch.cat((keep_values, keep_values))
        # update keep_indices to users/items+self.n_users
        keep_indices[1] += self.n_users
        all_indices = torch.cat((keep_indices, torch.flip(keep_indices, [0])), 1)
        self.masked_adj = torch.sparse.FloatTensor(all_indices, all_values, self.norm_adj.shape).to(self.device)

    def _normalize_adj_m(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        c_inv_sqrt = torch.pow(col_sum, -0.5)
        cols_inv_sqrt = c_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return values

    def get_edge_info(self):
        rows = torch.from_numpy(self.interaction_matrix.row)
        cols = torch.from_numpy(self.interaction_matrix.col)
        edges = torch.stack([rows, cols]).type(torch.LongTensor)
        # edge normalized values
        values = self._normalize_adj_m(edges, torch.Size((self.n_users, self.n_items)))
        return edges, values

    def forward(self, adj):
        user_emb = self.user_embedding.weight
        item_image_emb = self.image_trs(self.image_embedding.weight)
        item_text_emb = self.text_trs(self.text_embedding.weight)
        i_v_pos = self.i_pe(item_image_emb)
        i_t_pos = self.i_pe(item_text_emb)
        u_pos = self.u_pe(user_emb)
        alpha = 0.4
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

        ua_embeddings, ia_embeddings = self.forward(self.masked_adj)
        # ua_embeddings = self.u_pe(ua_embeddings)
        # ia_embeddings = self.i_pe(ia_embeddings)
        self.build_item_graph = False

        batch_mf_loss = self.ssm_loss(ua_embeddings, ia_embeddings, users, pos_items)
        # mf_v_loss, mf_t_loss = 0.0, 0.0
        # ua_embeddings = self.u_pe(ua_embeddings)
        # if self.t_feat is not None:
        #     text_feats = self.text_trs(self.text_embedding.weight)
        #     # text_feats = self.i_pe(text_feats)
        #     # mf_t_loss = self.bpr_loss(ua_embeddings[users], text_feats[pos_items], text_feats[neg_items])
        #     mf_t_loss = self.ssm_loss(ua_embeddings, text_feats, users, pos_items)
        # if self.v_feat is not None:
        #     image_feats = self.image_trs(self.image_embedding.weight)
        #     # image_feats = self.i_pe(image_feats)
        #     # mf_v_loss = self.bpr_loss(ua_embeddings[users], image_feats[pos_items], image_feats[neg_items])
        #     mf_v_loss = self.ssm_loss(ua_embeddings, image_feats, users, pos_items)
        # return mf_v_loss + mf_t_loss 
        return batch_mf_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

