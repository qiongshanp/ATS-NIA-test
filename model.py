import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import copy
import os
from dgl.nn.pytorch.conv.pnaconv import aggregate_max
import numpy as np
import scipy.sparse as sp
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.nn import GATConv,GATv2Conv,GCNConv, SAGEConv,SGConv,SSGConv, APPNP

def build_csr_adjacency_from_dgl(g, self_loop=False):
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()
    num_nodes = g.num_nodes()

    if self_loop:
        loop_index = np.arange(num_nodes)
        src = np.concatenate([src, loop_index])
        dst = np.concatenate([dst, loop_index])

    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    return adj.tocsr()

def build_adj_tensors(g):
    adj = build_csr_adjacency_from_dgl(g)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(g.number_of_nodes())
    adj[adj > 1] = 1
    return adj


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def trans_data(g, device):
    adj_np = build_adj_tensors(g.cpu())
    edge_index, edge_weight = utils.from_scipy_sparse_matrix(adj_np)
    features_tensor = g.ndata['feat'].double().float()
    labels = g.ndata['label'].to(device)
    data = Data(x=features_tensor, y=labels, edge_index=edge_index, num_nodes=features_tensor.shape[0], num_features=features_tensor.shape[1], n_class=g.ndata['label'].max().item() + 1)
    data = data.to(device)

    return data

class VictimModel():
    def __init__(self, in_feats, h_feats, num_classes, device, name):

        assert name in ['GCN', 'SGC', 'APPNP', 'GraphSAGE', 'GAT'], "GNN model not implement"
        if name == 'GCN':
            self.model = GCN(in_feats, h_feats, num_classes, 0.5)
        elif name == 'APPNP':
            self.model = APPNP(in_feats, h_feats, num_classes)
        elif name == 'GraphSAGE':
            self.model = GraphSAGE(in_feats, h_feats, num_classes)
        elif name == 'GAT':
            self.model = GAT(in_feats, h_feats, num_classes, 0.5, 0.5, heads=8)
        elif name == 'SGC':
            self.model = SGC(in_feats, h_feats, num_classes, 0.5, num_layers=2)

        self.model.to(device)


    def optimize(self, g, index_split, epochs, lr, patience, device, save_model_path, model_name, dataset):

        data = trans_data(g, device)

        self.model = self.model.to(device)

        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.0005)
        train_index, val_index, test_index = index_split['train_index'], index_split['val_index'], index_split['test_index']
        labels = g.ndata['label']

        self.model.train()
        best_val_acc = 0
        cnt = 0
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(data.x, data.edge_index)
            logp = F.log_softmax(logits, dim=1)
            loss = F.nll_loss(logp[train_index], labels[train_index])
            loss.backward()
            optimizer.step()

            self.model.eval()
            logits = self.model(data.x, data.edge_index)
            logp = F.log_softmax(logits, dim=1)
            acc_val = accuracy(logp[val_index], labels[val_index])
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                cnt = 0
            else:
                cnt += 1
            if cnt >= patience and epoch > 200:
                break
            del loss, logits
        torch.save(best_model_state_dict, os.path.join(save_model_path, model_name + '_' + dataset + '_checkpoint.pkl'),_use_new_zipfile_serialization=False)
        return best_model_state_dict, data


    def eval(self, g, index_split, save_model_path, model_name, dataset, data):
        self.model.load_state_dict(torch.load(os.path.join(save_model_path, model_name + '_' + dataset + '_checkpoint.pkl'), weights_only=False))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        labels = g.ndata['label']
        test_mask = g.ndata['test_mask']
        test_index = index_split['test_index']

        logits = self.model(data.x, data.edge_index)
        logp = F.log_softmax(logits, dim=1)

        val_acc = accuracy(logp[val_mask], labels[val_mask])
        test_acc = accuracy(logp[test_mask], labels[test_mask])
        train_acc = accuracy(logp[train_mask], labels[train_mask])
        acc = accuracy(logp, labels)
        print("Train accuracy {:.4%}".format(train_acc))
        print("Validate accuracy new {:.4%}".format( val_acc))
        print("Test accuracy {:.4%}".format(test_acc))
        print("ALL accuracy {:.4%}".format(acc))
        _, indices = torch.max(logits, dim=1)
        correct_in_test = (indices[test_index] == labels[test_index]).nonzero(as_tuple=True)[0]
        acc_idx = test_index[correct_in_test.cpu()]
        return test_acc, acc_idx


    def get_loss(self, g, node_index, device):
        data = trans_data(g, device)
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            pred_labels = torch.argmax(logits, dim=1)
            logit = self.model(data.x, data.edge_index)[node_index]
            logit = logit.reshape(1, -1)
            loss = F.cross_entropy(logit, g.ndata['label'][node_index].reshape(1))
            success_mask = (pred_labels[:-1] != g.ndata['label'][:-1])
            num_success = success_mask.sum().item()

        return loss.item(), num_success


    def get_reward(self, node_index, previous_g, current_g, device):
        prev_loss, _ = self.get_loss(previous_g, node_index, device)
        curr_loss, curr_success = self.get_loss(current_g, node_index, device)
        reward = (curr_loss - prev_loss)
        if curr_success > 0:
            reward += 1.0

        return reward, curr_success


class GCN(torch.nn.Module):

    def __init__(self, nfeat, hidden, classes, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(nfeat, hidden)
        self.conv2 = GCNConv(hidden, classes)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight != None:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.conv2(x, edge_index)

        return x


class GAT(torch.nn.Module):

    def __init__(self, nfeat, hidden, classes, dropout, att_dropout, heads):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(nfeat, hidden // heads, heads=heads, dropout=att_dropout)
        self.gat2 = GATConv(hidden, classes, dropout=att_dropout)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        return x



class SGC(torch.nn.Module):

    def __init__(self, feature, hidden, classes,dropout,num_layers):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.sgc = SGConv(feature, classes,K = num_layers)

    def forward(self,x, edge_index):

        x = self.sgc(x, edge_index)

        return x



class APPNP(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(APPNP, self).__init__()
        self.mlp = torch.nn.Linear(in_feats, num_classes)
        self.conv = dgl.nn.APPNPConv(k=3, alpha=0.5)

    def forward(self, g, in_feat):
        in_feat = self.mlp(in_feat)
        h = self.conv(g, in_feat)
        return h

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super().__init__()
        self.conv1 = dgl.nn.SAGEConv(in_feats, h_feats, aggregator_type='mean')
        self.conv2 = dgl.nn.SAGEConv(h_feats, num_classes, aggregator_type='mean')

    def forward(self, g, in_feats):
        h = self.conv1(g, in_feats)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

