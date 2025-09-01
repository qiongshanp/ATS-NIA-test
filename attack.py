import copy
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple
from tqdm import tqdm
import initial_feature_generate


PPOStep = namedtuple('PPOStep', ['action_log_prob', 'state_value', 'reward'])

class GumbelSoftmaxStraightThrough(torch.distributions.RelaxedOneHotCategorical):
    def __init__(self, temperature, logits=None, probs=None):
        super().__init__(temperature, logits=logits, probs=probs)

    def rsample(self, sample_shape=torch.Size()):
        y_soft = super().rsample(sample_shape)
        index = y_soft.max(-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        return (y_hard - y_soft).detach() + y_soft

    def log_prob_safe(self, y):
        eps = 1e-20
        log_p = torch.log(self.probs + eps)
        return (y * log_p).sum(dim=-1)



class Node_Generator(torch.nn.Module):
    def __init__(self, in_feats, h_feats):
        super(Node_Generator, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both')
        self.mlp = torch.nn.Sequential(
            nn.Linear(3 * h_feats, in_feats),
            nn.LeakyReLU(),
            nn.Linear(in_feats, in_feats)
        )
    def forward(self, g, node_index):
        h = F.leaky_relu(self.conv1(g, g.ndata['feat']))
        h = F.leaky_relu(self.conv2(g, h))
        graph_feat = torch.cat([h.mean(dim=0).flatten(), h.max(dim=0).values.flatten(), h[node_index].flatten()], dim=0)
        residual = self.mlp(graph_feat)
        residual = F.normalize(residual, dim=0)
        dist = torch.distributions.Normal(residual, torch.ones_like(residual) * 0.1)
        sample = dist.rsample()
        log_prob = dist.log_prob(sample).mean()
        return sample, residual, log_prob

class Edge_Sampler(torch.nn.Module):
    def __init__(self, in_feats, h_feats, alpha_n=1000000):
        super(Edge_Sampler, self).__init__()
        self.conv1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.conv2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both')
        self.regressor = torch.nn.Linear(in_feats + h_feats * 2, 1)
        self.activation = torch.nn.LeakyReLU()
        self.alpha_n = alpha_n
        torch.nn.init.xavier_normal_(self.regressor.weight)
        torch.nn.init.zeros_(self.regressor.bias)

    def forward(self, g, node_index, edge_set):
        num_candidate_dst = g.number_of_nodes() - 1 - len(edge_set)
        mask = [i for i in range(g.number_of_nodes() - 1) if i not in edge_set]

        if num_candidate_dst == 1:
            sample_ = torch.zeros(g.number_of_nodes() - 1).to(g.device)
            sample_[mask[0]] = 1.0
            log_prob = torch.tensor(0.0).to(g.device)
        else:
            node_neighbors_mask = torch.full((num_candidate_dst,), self.alpha_n, device=g.device)
            sampled_feature = g.ndata['feat'][-1].expand((num_candidate_dst, -1))
            h = self.activation(self.conv1(g, g.ndata['feat']))
            h = self.activation(self.conv2(g, h))
            target_node_embedding = h[node_index].expand((num_candidate_dst, -1))
            candidate_dst_h = h[mask]
            h = torch.cat((candidate_dst_h, sampled_feature, target_node_embedding), dim=1)

            logits = self.regressor(h).squeeze() + node_neighbors_mask
            dist = GumbelSoftmaxStraightThrough(temperature=1.0, logits=logits)
            sample = dist.rsample()
            log_prob = dist.log_prob_safe(sample)

            sample_ = torch.zeros(g.number_of_nodes() - 1).to(g.device)
            sample_[mask] = sample

        return sample_, log_prob


class Value_Predictor(torch.nn.Module):
    def __init__(self, in_feats, h_feats, n_class=7):
        super(Value_Predictor, self).__init__()
        self.gc1 = dgl.nn.GraphConv(in_feats, h_feats, norm='both')
        self.gc2 = dgl.nn.GraphConv(h_feats, h_feats, norm='both')
        self.regressor = torch.nn.Linear(h_feats, n_class)
        self.projector = torch.nn.Linear(h_feats + n_class, n_class)
        self.activation = torch.nn.LeakyReLU()
        self.n_class = n_class
    def forward(self, g, node_index, label):
        g, node_index = dgl.khop_in_subgraph(g, node_index, 2)
        h1 = self.activation(self.gc1(g, g.ndata['feat']))
        h1 = self.activation(self.gc2(g, h1))
        logits = F.log_softmax(self.regressor(h1), dim=-1)
        h = torch.cat((h1[node_index], logits[node_index]), dim=-1)
        h = self.projector(h)
        return F.cross_entropy(h.reshape(1, self.n_class), label.reshape(1))

def inject_node(g, feat):
    nid = g.num_nodes()
    g = dgl.add_nodes(g, 1, {'feat': feat.reshape(1, -1)})
    g = dgl.add_edges(g, nid, nid)
    return g

def wire_edge(g, dst):
    g = dgl.add_edges(g, torch.tensor([dst, g.number_of_nodes() - 1]).to(g.device),
                         torch.tensor([g.number_of_nodes() - 1, dst]).to(g.device))
    return g


def compute_homophily_loss(final_feat, subg, target_idx):
    neighbor_ids = subg.successors(target_idx)
    if len(neighbor_ids) > 0:
        neighbor_feats = subg.ndata['feat'][neighbor_ids]
        neighbor_mean = neighbor_feats.mean(dim=0)
        return 1 - F.cosine_similarity(final_feat, neighbor_mean, dim=0).mean()
    else:
        return 1 - F.cosine_similarity(final_feat, subg.ndata['feat'][target_idx], dim=0).mean()


class Injectattack():
    def __init__(self,graphdata, in_dim, hid_dim, out_dim, args, device):
        self.args = args
        self.node_generator = Node_Generator(in_dim, hid_dim * 2).to(device)
        self.edge_sampler = Edge_Sampler(in_dim, hid_dim * 2).to(device)
        self.value_predictor = Value_Predictor(in_dim, hid_dim * 2, out_dim).to(device)
        self.optimizer = optim.Adam(list(self.node_generator.parameters()) + list(self.edge_sampler.parameters()) + list(self.value_predictor.parameters()), lr=1e-4)
        self.eps = np.finfo(np.float32).eps.item()
        self.initial_features = {}

    def finish_episode_ppo(self, step, device, ppo_buffer, feature_loss, clip_eps=0.2, entropy_coef=0.01):
        R = 0
        returns = []
        for step_tuple in ppo_buffer[::-1]:
            R = step_tuple.reward + self.args.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        ep_reward = returns.sum().item()
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        log_probs = torch.stack([x.action_log_prob for x in ppo_buffer])
        values = torch.stack([x.state_value for x in ppo_buffer])
        old_log_probs = log_probs.detach()
        advantages = returns - values.detach().squeeze()
        ratio = torch.exp(log_probs - old_log_probs)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        value_loss = F.smooth_l1_loss(values.squeeze(), returns.squeeze())
        entropy_bonus = -log_probs.mean()
        loss = policy_loss + value_loss + feature_loss + entropy_coef * entropy_bonus
        loss.backward()
        if step % self.args.accumulation_step == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            step = 0
        return step, ep_reward



    def Attack_generation(self, g, attack_node, surrogate_model, final_accs, device):
        lowest_acc = np.inf
        cnt = 0
        best_solution = None
        best_solved_set = None
        feature_gen = {}
        attacked_nodes = set()

        graph = copy.deepcopy(g)
        for e in range(self.args.epochs):
            total_success = 0
            step = 0
            solved_set = []
            unsolved_set = []
            accumulated_ep_reward = 0
            iters = 0
            solution = {}
            sum_nodes = 0
            sum_edges = 0
            valid_attack_nodes = [n for n in attack_node if n not in attacked_nodes]
            pbar = tqdm(valid_attack_nodes) if not self.args.verbose else valid_attack_nodes

            for node_index in pbar:
                if node_index in attacked_nodes: continue
                original_node_index = node_index
                if self.args.khop_edge != 0:
                    subg, induced_nodes = dgl.khop_in_subgraph(graph, node_index, self.args.khop_edge)
                    new_node_indices = subg.nodes()
                    original_node_indices = subg.ndata[dgl.NID]
                else:
                    subg = graph
                    induced_nodes = node_index
                    original_node_indices = torch.tensor([node_index])

                ppo_buffer = []
                feature_loss = 0
                success = 0
                step += 1
                iters += 1
                node_attribute_buffer = []
                link_buffer = []
                node_count = 0
                edge_count = 0
                edge_set = []
                for node_epochs in range(self.args.node_budget):
                    node_count += 1
                    if node_epochs > 0:
                        original_node_indices[-1] = g.num_nodes() - 1 + node_epochs + sum_nodes
                    if e == 0:
                        feature = initial_feature_generate.optimize_features_nn(g.ndata['feat'], original_node_index, num_injected=1, epochs=100, device=device)
                        feature_gen[original_node_index.item()] = feature.detach()
                        action_log_prob = torch.tensor(0.0).to(device)
                    else:
                        reference_feat = feature_gen[original_node_index.item()]
                        residual, _, action_log_prob = self.node_generator(subg, induced_nodes)
                        final_feat = reference_feat + residual
                        imitation_loss = F.mse_loss(final_feat, reference_feat.detach())
                        homophily_loss = compute_homophily_loss(final_feat, subg, induced_nodes)
                        feature_loss += (imitation_loss * 1.0 + homophily_loss * 0.5) / self.args.node_budget
                        feature = final_feat

                    if e > 0:
                        node_attribute_buffer.append(feature.detach().cpu())
                        link_buffer.append([])

                    for i in range(self.args.edge_budget):
                        if i == 0:
                            previous_graph = subg.clone()
                            subg = inject_node(subg, feature)
                            subg = wire_edge(subg, induced_nodes)

                            edge_set = [induced_nodes]
                            reward, success = surrogate_model.get_reward(induced_nodes, previous_graph, subg, device)
                            state_value = self.value_predictor(subg, induced_nodes, subg.ndata['label'][induced_nodes])
                            ppo_buffer.append(PPOStep(action_log_prob, state_value, reward))
                            dst = induced_nodes
                            edge_count += 1
                            if e > 0:
                                link_buffer[-1].append(dst.item())
                        else:
                            edge_dist, edge_log_prob = self.edge_sampler(subg, induced_nodes, edge_set)
                            state_value = self.value_predictor(subg, induced_nodes, subg.ndata['label'][induced_nodes])
                            ppo_buffer.append(PPOStep(edge_log_prob + action_log_prob, state_value, reward))
                            previous_graph = subg.clone()
                            dst = (edge_dist == 1).nonzero(as_tuple=True)[0][0]
                            with subg.local_scope():
                                subg = wire_edge(subg, dst)
                                reward, add_success = surrogate_model.get_reward(
                                    induced_nodes, previous_graph, subg, device)
                            if add_success > success:
                                edge_count += 1
                                success = add_success
                                subg = wire_edge(subg, dst)
                                if e > 0:
                                    link_buffer[-1].append(dst.item())
                            edge_set.append(dst)
                            ppo_buffer[-1] = ppo_buffer[-1]._replace(reward=reward)
                            original_node_indices = subg.ndata[dgl.NID]
                        if len(edge_set) + 1 == subg.number_of_nodes(): break
                        if success >= len(new_node_indices): break


                    if success > 1:
                        total_success += 1
                        break


                if success > 1 and e > 0:
                    sum_nodes += node_count
                    sum_edges += edge_count
                    solved_set.append(original_node_index.item())
                    node_set = [[original_node_indices[i].item() for i in sublist] for sublist in link_buffer]
                    solution[original_node_index.item()] = {'node_feat': torch.stack(node_attribute_buffer), 'edges': node_set}
                    for n in original_node_indices.tolist():
                        attacked_nodes.add(n)
                else:
                    unsolved_set.append(original_node_index.item())

                step, ep_reward = self.finish_episode_ppo(step, device, ppo_buffer, feature_loss)
                accumulated_ep_reward += ep_reward
                total_test_nodes = sum(g.ndata['test_mask'])
                current_acc = (len(attack_node) - total_success) * 100 / total_test_nodes
                if not self.args.verbose:
                    pbar.set_description('Epoch: {}, Current Accuracy: {:.2f} ({}/{}), Episode Reward: {:.2f}'.format(e + 1, current_acc, len(attack_node) - total_success, len(attack_node), accumulated_ep_reward / iters))

            if e > 0:
                if current_acc < lowest_acc:
                    lowest_acc = current_acc
                    cnt = 0
                    best_solution = solution
                    best_solved_set = solved_set
                else:
                    cnt += 1
            if cnt == self.args.patience:
                print('Early Stopping. Current Accuracy is {:.2f}.'.format(lowest_acc))
                if not self.args.verbose:
                    print('Solution saved to {}'.format(self.args.save_dir))
                    import pickle
                    with open(f'{self.args.save_dir}' + f'{self.args.dataset}.pickle', 'wb') as handle:
                        pickle.dump(best_solution, handle, protocol=pickle.HIGHEST_PROTOCOL)
                final_accs.append(lowest_acc)
                break

        return best_solution, best_solved_set
