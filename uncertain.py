import torch
import torch.nn as nn
import dgl.function as fn
import numpy as np
import torch.nn.functional as F



class Bayesian_Network(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, T, theta, device):
        super(Bayesian_Network, self).__init__()
        self.layer1 = nn.Parameter(torch.Tensor(in_dim, hid_dim))
        self.layer2 = nn.Parameter(torch.Tensor(hid_dim, out_dim))
        self.relu = nn.ReLU()
        self.device = device
        self.T = T
        self.theta = theta
        self.mask1 = (torch.rand(T, in_dim, hid_dim) <= theta).to(device)
        self.mask2 = (torch.rand(T, hid_dim, out_dim) <= theta).to(device)
        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.layer1, gain=gain)
        nn.init.xavier_uniform_(self.layer2, gain=gain)

    def forward(self, g):
        result=[]
        for t in range(self.T):
            with g.local_scope():
                g.update_all(fn.u_mul_e('feat', 'edge_weight', 'm'), fn.sum('m', 'h'))
                self.layer1.to(self.device)
                self.layer2.to(self.device)
                w1 = torch.mul(self.layer1, self.mask1[t])
                g.ndata['h'] = self.relu(torch.mm(g.ndata['h'], w1))
                g.update_all(fn.u_mul_e('h', 'edge_weight', 'm'), fn.sum('m', 'o'))
                w2 = torch.mul(self.layer2, self.mask2[t])
                y = torch.mm(g.ndata['o'], w2)
                result.append(y)
        return result

    def optimize(self, g, lr):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        label = g.ndata['label']
        label_idx = (label >= 0)

        degree = g.in_degrees()
        degree = torch.pow(degree, -0.5)
        src, dst = g.edges()
        g.edata['edge_weight'] = degree[src] * degree[dst]

        for epoch in range(500):
            output = self(g)
            loss = 0
            for y_hat in output:
                loss += loss_fn(y_hat[label_idx], label[label_idx])
            loss = loss / self.T + (1-self.theta) / self.T * ((self.layer1 ** 2).mean() + (self.layer2 ** 2).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            output = self(g)
            output = torch.stack(output, dim=0)
            output = F.softmax(output, dim=2)
            uncertainty = torch.var(output, dim=0).mean(dim=1)
            return  uncertainty

def get_uncertainty_score(g, in_dim, hid_dim, out_dim, device, T=20, theta=0.1, lr=0.01):
    model = Bayesian_Network(in_dim, hid_dim, out_dim, T, theta, device)
    uncertainty = model.optimize(g, lr)
    return uncertainty


def node_select(mode, g, ratio, test_idx, device, in_dim, hid_dim, out_dim, args):
    tensor = torch.zeros(g.num_nodes(), dtype=torch.bool).to(device)
    tensor[test_idx] = True

    if mode == "uncertainty":
        bayesian_network = Bayesian_Network(in_dim, hid_dim, out_dim, args.T, args.theta, device).to(device)
        uncertainty = bayesian_network.optimize(g, args.lr).to(device)
        unc_a = torch.where(tensor, uncertainty, 0)
        value, idx_a = torch.sort(unc_a, descending=True)
    elif mode == "degree":
        degree = g.in_degrees() + g.out_degrees()
        unc_a = torch.where(tensor, degree, torch.max(degree) + 1)
        value, idx_a = torch.sort(unc_a, descending=False)
    elif mode == "random":
        shuffled_indices = np.random.permutation(np.arange(len(test_idx)))
        idx_a = test_idx[shuffled_indices]
    else:
        raise NotImplementedError

    num_nodes_to_select = max(1, int(ratio * g.num_nodes()))
    idx_node = idx_a[:num_nodes_to_select ]
    return idx_node

def compute_degree_centrality(g):
    deg = g.in_degrees() + g.out_degrees()
    deg = deg.float()
    deg = (deg - deg.min()) / (deg.max() - deg.min() + 1e-6)
    return deg

def composite_score_select(g, acc_idx, in_dim, hid_dim, out_dim, args, device, ratio, lambda_=0.6):
    acc_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).to(device)
    acc_mask[acc_idx] = True

    uncertainty = get_uncertainty_score(g, in_dim, hid_dim, out_dim, device, args.T, args.theta, args.lr)
    centrality = compute_degree_centrality(g).to(device)

    score = lambda_ * uncertainty + (1 - lambda_) * centrality
    score = torch.where(acc_mask, score, torch.tensor(-1e9, device=device))

    top_k = max(1, int(ratio * g.num_nodes()))
    _, idx = torch.topk(score, top_k)
    return idx


def cascading_score_select(g, prev_nodes, acc_idx, in_dim, hid_dim, out_dim, args, device, ratio, lambda_=0.6):
    num_to_select = max(1, int(ratio * g.num_nodes()))
    uncertainty = get_uncertainty_score(g, in_dim, hid_dim, out_dim, device, args.T, args.theta, args.lr)
    centrality = compute_degree_centrality(g).to(device)
    score = lambda_ * uncertainty + (1 - lambda_) * centrality

    acc_mask = torch.zeros(g.num_nodes(), dtype=torch.bool).to(device)
    acc_mask[acc_idx] = True
    inherited_from_map = {}
    for n in prev_nodes:
        neighbors = set(g.successors(n).tolist() + g.predecessors(n).tolist())
        for nbr in neighbors:
            if nbr not in inherited_from_map:
                inherited_from_map[nbr] = set()
            inherited_from_map[nbr].add(n)
    candidate = list(inherited_from_map.keys())
    candidate = torch.tensor(candidate).to(device)

    selected = []
    if len(candidate) > 0:
        local_score = score[candidate]
        top_k = min(len(candidate), num_to_select)
        _, idx_local = torch.topk(local_score, top_k)
        selected = candidate[idx_local].tolist()

    if len(selected) < num_to_select:
        already_selected = set(selected + list(prev_nodes))
        remain_mask = acc_mask.clone()
        remain_mask[list(already_selected)] = False
        score = torch.where(remain_mask, score, torch.tensor(-1e9, device=device))
        num_add = num_to_select - len(selected)
        _, idx_supp = torch.topk(score, num_add)
        selected += idx_supp.tolist()

    return torch.tensor(selected, dtype=torch.long).to(device), inherited_from_map

