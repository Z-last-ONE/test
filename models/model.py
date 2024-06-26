# coding: utf-8
#
#
# user-graph need to be generated by the following script
# tools/generate-u-u-matrix.py
import os

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender


class MODEL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MODEL, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_text_weight = config['mm_text_weight']
        self.alpha_item = config['alpha_item']
        self.alpha_sem = config['alpha_sem']
        self.alpha_image = config['alpha_image']
        self.alpha_high_item = config['alpha_high_item']
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = dataset
        # self.construction = 'weighted_max'
        self.construction = 'cat'
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat = 128
        self.embedding_size = self.dim_latent * 3
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        # self.MLP_1 = nn.Sequential(
        #     nn.Linear(self.dim_latent, self.dim_latent),
        #     nn.Tanh(),
        #     nn.Linear(self.dim_latent, 1, bias=False)
        # )
        # self.MLP_2 = nn.Sequential(
        #     nn.Linear(self.dim_latent, self.dim_latent),
        #     nn.Tanh(),
        #     nn.Linear(self.dim_latent, 1, bias=False)
        # )
        # self.MLP_3 = nn.Sequential(
        #     nn.Linear(self.dim_latent, self.dim_latent),
        #     nn.Tanh(),
        #     nn.Linear(self.dim_latent, 1, bias=False)
        # )
        # for layer in self.MLP_1.children():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        # for layer in self.MLP_2.children():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)
        # for layer in self.MLP_3.children():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_normal_(layer.weight)

        self.mm_adj = None
        # load data
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])

        # self.user_prof = torch.from_numpy(np.load('user_prof.npy')).type(
        #     torch.FloatTensor).to(self.device)
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.embedding_size, (self.user_prof.shape[1] + self.embedding_size) // 2),
        #     nn.LeakyReLU(),
        #     nn.Linear((self.user_prof.shape[1] + self.embedding_size) // 2, self.user_prof.shape[1])
        # )

        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()
        self.HL = np.load(os.path.join(dataset_path, config['high_item_graph_dict_file']),
                          allow_pickle=True).item()
        self.split_HL_edge()
        if config['use_weighting']:
            self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),
                                           allow_pickle=True).item()
        self.u_feat = None
        if config['user_feature']:
            self.u_feat = torch.from_numpy(
                np.load(f'{dataset_path}/{config["user_feature_file"]}', allow_pickle=True)).type(torch.FloatTensor).to(
                self.device)
        # load features and adj
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
            self.image_adj = image_adj

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
            self.text_adj = text_adj

        if self.i2t_feat is not None:
            self.i2t_embedding = nn.Embedding.from_pretrained(self.i2t_feat, freeze=False)
            _, i2t_adj = self.get_knn_adj_mat(self.i2t_embedding.weight.detach())
            self.i2t_adj = i2t_adj

        if config['use_weighting']:
            _, co_adj = self.get_co_occurrence_item()
            self.co_adj = co_adj
        # load inters
        self.train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(self.train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.edge_index2 = None
        if True:
            edge_head = []
            edge_rear = []
            '''
            # user edge and item edge
            for key, value in self.user_graph_dict.items():
                neighbors = value[0][:10]
                for neighbor in neighbors:
                    edge_head.append(key)
                    edge_rear.append(neighbor)
                    # TODO
                    edge_head.append(neighbor)
                    edge_rear.append(key)
                    '''
            for key, value in self.item_graph_dict.items():
                neighbors = value[0][:10]
                for neighbor in neighbors:
                    edge_head.append(key + num_user)
                    edge_rear.append(neighbor + num_user)
                    # TODO
                    edge_head.append(neighbor + num_user)
                    edge_rear.append(key + num_user)
            result = np.vstack([edge_head, edge_rear]).transpose()
            edge = torch.LongTensor(result).t().contiguous().to(self.device)
            self.edge_index = torch.cat((self.edge_index, edge), dim=1)
            self.edge_index2 = edge

        # weight
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 3, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        # self.weight_i = nn.Parameter(nn.init.xavier_normal_(
        #     torch.tensor(np.random.randn(self.num_item, 2, 1), dtype=torch.float32, requires_grad=True)))
        # self.weight_i.data = F.softmax(self.weight_i, dim=1)

        # self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)
        # gcn module
        if self.v_feat is not None:
            self.v_gcn = GCN(config, self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.v_feat, u_feat=self.u_feat)  # 256)
        if self.t_feat is not None:
            self.t_gcn = GCN(config, self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                             device=self.device, features=self.t_feat, u_feat=self.u_feat)

        if self.i2t_feat is not None:
            self.i2t_gcn = GCN(config, self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                               device=self.device, features=self.i2t_feat, u_feat=self.u_feat)

        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

    def get_co_occurrence_item(self):
        graph_co = self.item_graph_dict
        # indices0 = torch.arange(len(graph_co)).to(self.device)
        # indices0 = torch.unsqueeze(indices0, 1)
        # indices0 = indices0.expand(-1, self.knn_k)
        indices = []
        result = []
        for indx, v in graph_co.items():
            if len(v[0]) >= 10:
                indices.append(np.full(10, indx))
            else:
                length = len(v[0])
                indices.append(np.full(length, indx))
            result.append(v[0][:10])
        indices = torch.IntTensor(np.concatenate(indices)).to(self.device)
        result = torch.IntTensor(np.concatenate(result)).to(self.device)

        indices = torch.stack((torch.flatten(indices), torch.flatten(result)), 0).to(torch.int64).to(self.device)
        adj_size = torch.Size([len(graph_co), len(graph_co)])
        return indices, self.compute_normalized_laplacian(indices, adj_size)

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

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users

        v_rep, self.v_preference = self.v_gcn(self.edge_index, self.v_feat, item_edge=self.edge_index2)
        t_rep, self.t_preference = self.t_gcn(self.edge_index, self.t_feat, item_edge=self.edge_index2)
        i2t_rep, self.i2t_preference = self.i2t_gcn(self.edge_index, self.i2t_feat, item_edge=self.edge_index2)
        representation = torch.cat((v_rep, t_rep, i2t_rep), dim=1)

        self.v_rep = torch.unsqueeze(v_rep, 2)
        self.t_rep = torch.unsqueeze(t_rep, 2)
        self.i2t_rep = torch.unsqueeze(i2t_rep, 2)
        user_rep = torch.cat((self.v_rep[:self.num_user], self.t_rep[:self.num_user], self.i2t_rep[:self.num_user]),
                             dim=2)
        ##########gate###########################

        # score = torch.cat(
        #     (self.MLP_1(v_rep[:self.num_user]), self.MLP_2(t_rep[:self.num_user]), self.MLP_3(i2t_rep[:self.num_user])),
        #     dim=-1)
        # score = nn.Softmax(dim=-1)(score)
        # score = score.unsqueeze(-1).transpose(1, 2)
        # global_avg_pooling, _ = torch.max(user_rep, dim=1, keepdim=True)
        # global_avg_pooling = self.MLP_1(global_avg_pooling)
        # global_avg_pooling = nn.ReLU()(global_avg_pooling)
        # global_avg_pooling = self.MLP_2(global_avg_pooling)
        # score = nn.Sigmoid()(score)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        # user_rep = score * user_rep
        user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)
        self.user_rep = user_rep
        item_rep = representation[self.num_user:]

        ############################################ multi-modal information aggregation
        ht = item_rep
        for i in range(self.n_layers):
            ht = torch.sparse.mm(self.text_adj, ht)
        hi = item_rep
        for i in range(self.n_layers):
            hi = torch.sparse.mm(self.image_adj, hi)
        hc = item_rep
        for i in range(self.n_layers):
            hc = torch.sparse.mm(self.co_adj, hc)
        hit = item_rep
        for i in range(self.n_layers):
            hit = torch.sparse.mm(self.i2t_adj, hit)
        hhl = item_rep
        for i in range(self.n_layers):
            hhl = torch.sparse.mm(self.HL_adj, hhl)

        h = self.mm_text_weight * ht + \
            self.alpha_sem * hit + \
            self.alpha_high_item * hhl + \
            self.alpha_item * hc + \
            self.alpha_image * hi
        # alpha = self.mm_text_weight
        # h = (1 - alpha) * ht + (alpha / 4) * hi + (alpha / 4) * hc + (alpha / 4) * hit + (alpha / 4) * hhl
        h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)

        user_rep = user_rep + h_u1
        item_rep = item_rep + h
        all_embedding = torch.cat((user_rep, item_rep), dim=0)

        # '''
        # item_user_embeds = torch.sparse.mm(self.R, h)
        # item_embeds = torch.cat([item_user_embeds, h], dim=0)
        # att_common = self.query_common(item_embeds)
        # weight_common = self.softmax(att_common)
        # image_prefer = self.gate_item_prefer(all_embedding)
        # sep_image_embeds = item_embeds - weight_common
        #
        # sep_image_embeds = torch.multiply(image_prefer, sep_image_embeds)
        # side_embeds = (sep_image_embeds + weight_common) / 2
        # '''

        self.result_embed = all_embedding
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores

    def split_HL_edge(self):
        s = []
        e = []
        for k, v in self.HL.items():
            if v < 3:
                continue
            for i in range(len(k)):
                for j in range(i + 1, len(k)):
                    s.append(k[i])
                    e.append(k[j])
                    # TODO 双向
                    s.append(k[j])
                    e.append(k[i])
        indices = torch.IntTensor(s).to(self.device)
        result = torch.IntTensor(e).to(self.device)

        indices = torch.stack((torch.flatten(indices), torch.flatten(result)), 0).to(torch.int64).to(self.device)
        adj_size = torch.Size([self.num_item, self.num_item])
        self.HL_adj = self.compute_normalized_laplacian(indices, adj_size)

    # def _reconstruction(self, user_rep, user):
    #     enc_embeds = user_rep[user]
    #     prf_embeds = self.user_prof[user]
    #     enc_embeds = self.mlp(enc_embeds)
    #     recon_loss = self.ssl_con_loss(enc_embeds, prf_embeds, 0.5)
    #     return recon_loss
    #
    # def ssl_con_loss(self, x, y, temp=1.0):
    #     x = F.normalize(x)
    #     y = F.normalize(y)
    #     mole = torch.exp(torch.sum(x * y, dim=1) / temp)
    #     deno = torch.sum(torch.exp(x @ y.T / temp), dim=1)
    #     return -torch.log(mole / (deno + 1e-8) + 1e-8).mean()

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_embedding_loss_i2t = (self.i2t_preference[user] ** 2).mean() if self.i2t_preference is not None else 0.0

        # reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t + reg_embedding_loss_i2t)
        # if self.construction == 'weighted_sum':
        #     reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        #     reg_loss += self.reg_weight * (self.weight_i ** 2).mean()
        # elif self.construction == 'cat':
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        # elif self.construction == 'cat_mlp':
        #     reg_loss += self.reg_weight * (self.MLP_user.weight ** 2).mean()

        # recon_loss = 0.1 * self._reconstruction(self.user_rep, user)

        return loss_value + reg_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.train_interactions.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            # norm_adj = adj.dot(d_mat_inv)
            # print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        # norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class Item_Graph_sample(torch.nn.Module):
    def __init__(self, num_item, aggr_mode, dim_latent):
        super(Item_Graph_sample, self).__init__()
        self.num_item = num_item
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, item_graph, item_matrix):
        index = item_graph
        u_features = features[index]
        user_matrix = item_matrix.unsqueeze(1)
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, config, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None, u_feat=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device
        self.config = config
        self.fW = nn.Parameter(torch.Tensor(3))

        if self.dim_latent:
            if config['user_feature']:
                self.preference = nn.Parameter(u_feat, requires_grad=True).to(self.device)
                self.user_trs = nn.Linear(u_feat.shape[1], self.dim_latent)
            else:
                self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                    np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                    gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            # self.conv_embed_2 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index, features, item_edge=None):
        # feat1 = self.MLP(features)
        # act1 = F.leaky_relu(feat1)
        # act2 = nn.Softmax()(act1)
        # temp_features = self.MLP_1(act2) if self.dim_latent else features
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        if self.config['user_feature']:
            preference = self.user_trs(self.preference)
            x = torch.cat((preference, temp_features), dim=0).to(self.device)

        else:
            x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)

        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)
        x_hat = h + x + h_1
        # if item_edge is not None:
        #     h_2 = self.conv_embed_2(x, item_edge)  # equation 1
        #     h_21 = self.conv_embed_2(h_2, item_edge)
        #     x_hat2 = h_2 + x + h_21
        #     x_hat = x_hat + x_hat2
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        # pdb.set_trace()
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
            # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        # pdb.set_trace()
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            # pdb.set_trace()
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
