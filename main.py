import argparse
import time
from data import load_data, setup_seed
from model import VictimModel, trans_data
from uncertain import *
from attack import Injectattack
from new_graph import *
from torch.nn.functional import cosine_similarity
import itertools
import os


def main(args):
    for rep in range(args.n_times):
        print('Currently runing at \033[95m{}\033[0m round'.format(str(rep)))
        setup_seed(args.seed)
        graphdata = load_data(args.dataset,device)
        g, index_split = graphdata.g, graphdata.index_split
        in_dim = g.ndata['feat'].shape[1]
        hid_dim = args.hid_dim
        out_dim = max(g.ndata['label']).item() + 1


        print('dataset nodes num: {}, train num: {} val num: {}, test num: {}'.format(g.ndata['feat'].shape[0],index_split['train_index'].shape[0],index_split['val_index'].shape[0],index_split['test_index'].shape[0]))
        save_surrogate_model_path = 'surrogate_model/'
        save_victim_model_path = 'victim_model/'

        stage_ratio = args.stage_ratio
        stage_ratio[-1] = 1.0 - sum(stage_ratio[:-1])

        print('======= Budget Overview =======')
        print('Dataset: \033[95m{}\033[0m'.format(args.dataset))
        print('VictimModel: \033[95m{}\033[0m, Surrogatemodel: \033[95m{}\033[0m'.format(args.victim_model, args.surrogate_model))
        print('Number of nodes: \033[95m{:d}\033[0m, Feature dimension: \033[95m{:d}\033[0m, Label category: \033[95m{:d}\033[0m'.format(g.num_nodes(), in_dim, out_dim))
        print('Ratio: \033[95m{:.2f}\033[0m'.format(args.ratio))
        print('Node Budget: \033[95m{:d}\033[0m'.format(args.node_budget))
        print('Edge Budget: \033[95m{:d}\033[0m'.format(args.edge_budget))
        print('\033[95m{}\033[0m Feature {}: \033[95m{:.2f}\033[0m'.format('Discrete' if graphdata.discrete_feat else 'Continuous', 'Budget' if graphdata.discrete_feat else 'Mass', args.feature_budget * graphdata.feature_budget))
        print('===============================')

        surrogate_model = VictimModel(in_dim, hid_dim, out_dim, device, args.surrogate_model)
        best_model_state_dict, data = surrogate_model.optimize(g, index_split, args.epochs, args.lr, args.patience, device, save_surrogate_model_path, args.surrogate_model, args.dataset)
        Before_acc_surrogate, acc_idx_surrogate = surrogate_model.eval(g, index_split, save_surrogate_model_path, args.surrogate_model, args.dataset, data)
        print('Before_acc_surrogate:\033[95m{:.4f}\033[0m'.format(Before_acc_surrogate * 100))


        newgraph = copy.deepcopy(g)
        ''' Node injection attack'''
        start_time = time.time()
        node_attack = Injectattack(graphdata, in_dim, hid_dim, out_dim, args, device)
        prev_attacked_nodes = set()
        target_to_injected_nodes = {}
        start_injected_id = newgraph.number_of_nodes()

        for a in range(args.update):
            print('Currently update at \033[95m{}\033[0m round'.format(a))
            if args.update != 1:
                ratio = args.ratio * stage_ratio[a]
            else:
                ratio = args.ratio


            if a == 0:
                attack_node = composite_score_select(newgraph, acc_idx_surrogate, in_dim, hid_dim, out_dim, args, device, ratio, lambda_=1)
            else:
                attack_node = composite_score_select(newgraph, acc_idx_surrogate, in_dim, hid_dim, out_dim, args,
                                                     device, ratio, lambda_ = 1 - (a / (args.update - 1)) * args.alpha)

            solution, solved_set = node_attack.Attack_generation(newgraph, attack_node, surrogate_model, final_accs, device)
            newgraph = graph_generation(newgraph, solution, device)
            prev_attacked_nodes.update(solution.keys())
            for target_node, inject_data in solution.items():
                num_new_nodes = inject_data['node_feat'].shape[0]
                injected_node_ids = list(range(start_injected_id, start_injected_id + num_new_nodes))
                target_to_injected_nodes[target_node] = injected_node_ids
                start_injected_id += num_new_nodes
            print('target_to_injected_nodes', target_to_injected_nodes)


            data_newgraph = trans_data(newgraph, device)
            After_acc_surrogate, acc_idx_surrogate = surrogate_model.eval(newgraph, index_split, save_surrogate_model_path, args.surrogate_model, args.dataset, data_newgraph)
        After_acc_victim = []
        Before_acc_victim = []
        victimmodel = args.victim_model
        victim_model = VictimModel(in_dim, hid_dim, out_dim, device, victimmodel)
        best_model_state_dict2, data2 = victim_model.optimize(g, index_split, args.epochs, args.lr, args.patience,device, save_victim_model_path, victimmodel,args.dataset)
        acc_victim1, acc_idx_victim = victim_model.eval(g, index_split, save_victim_model_path, victimmodel,args.dataset, data2)
        Before_acc_victim.append(acc_victim1)
        acc_victim2, acc_idx_victim = victim_model.eval(newgraph, index_split, save_victim_model_path, victimmodel, args.dataset, data_newgraph)
        After_acc_victim.append(acc_victim2)
        end_time = time.time()
        print('Before_acc_victim:', ['{:.4f}'.format(acc * 100) for acc in Before_acc_victim])
        print(">> Finish attack, cost {:.4f}s.".format(end_time - start_time))

        print('======= Budget Overview =======')
        print('Dataset: \033[95m{}\033[0m, VictimModel: \033[95m{}\033[0m, Surrogatemodel: \033[95m{}\033[0m'.format(args.dataset, args.victim_model, args.surrogate_model))
        print('Number of nodes: \033[95m{:d}\033[0m, Feature dimension: \033[95m{:d}\033[0m, Label category: \033[95m{:d}\033[0m'.format(g.num_nodes(), in_dim, out_dim))
        print('Attack mode: \033[95m{}\033[0m, Update number: \033[95m{}\033[0m, Ratio: \033[95m{:.2f}\033[0m'.format(args.attack_mode, args.update, args.ratio))
        print('Node Budget: \033[95m{:d}\033[0m, Edge Budget: \033[95m{:d}\033[0m'.format(args.node_budget, args.edge_budget))
        print('\033[95m{}\033[0m Feature {}: \033[95m{:.2f}\033[0m'.format('Discrete' if graphdata.discrete_feat else 'Continuous', 'Budget' if graphdata.discrete_feat else 'Mass', args.feature_budget * graphdata.feature_budget))
        print("Finish attack, cost \033[95m{:.4f}\033[0ms.".format(end_time - start_time))
        print('Before_Acc_surrogate \033[95m{}\033[0m:\033[95m{:.4f}\033[0m, After_Acc_surrogate:\033[95m{:.4f}\033[0m'.format(args.surrogate_model, Before_acc_surrogate * 100, After_acc_surrogate * 100))
        print('===============================')
        print("======= Victim Model Accuracy Summary =======")
        for i, model in enumerate(args.victim_model):
            print(f"{model}: Before = \033[95m{Before_acc_victim[i] * 100:.2f}%\033[0m, After = \033[91m{After_acc_victim[i] * 100:.2f}%\033[0m")



if __name__ == '__main__':

    setup_seed(123)
    parser = argparse.ArgumentParser(description='MNIA')
    parser.add_argument('--dataset', type=str, default='citeseer', choices=['citeseer', 'cora', 'pubmed', 'wiki_cs', 'co_computer', 'co_photo', 'ogbproducts'], help='dataset to attack')
    parser.add_argument('--attack_mode', type=str, default='uncertainty', choices=['uncertainty', 'random', 'degree'], help='attack node select mode')
    parser.add_argument('--seed', type=int, default=123, help='random seed')

    parser.add_argument('--hid_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--patience', type=int, default=10, help='early stop patience')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--ratio', type=float, default=0.03, help='node of top ratio uncertainty are attacked')
    parser.add_argument('--update', type=int, default=3, help='update number')
    parser.add_argument('--stage_ratio', type=float, nargs='*',default=[0.33, 0.33, 0.33], help='ratio for each injection stage (sum to 1.0)')
    parser.add_argument('--alpha', type=float,default=1, help='alpha parameter')

    parser.add_argument('--models', type=str, nargs='*', default=['GCN'], choices=['GCN', 'SGC', 'GAT'])
    parser.add_argument('--surrogate_model', type=str, nargs='*', default='GCN', choices=['GCN', 'SGC', 'GAT'])
    parser.add_argument('--victim_model', type=str, nargs='*', default=['GCN'],choices=['GCN', 'SGC', 'GAT'])
    parser.add_argument('--n_times', type=int, default=1, help='time to run')
    parser.add_argument('--device', type=int, default=0, help='device ID for GPU')

    parser.add_argument('--T', type=int, default=20, help='sampling times of Bayesian Network')
    parser.add_argument('--theta', type=float, default=0.1, help='bernoulli parameter of Bayesian Network')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--khop_feat', type=int, default=2, help='order of subgraphs to generate node features')
    parser.add_argument('--khop_edge', type=int, default=2, help='order of subgraphs to wire node, 0 for full graph')
    parser.add_argument('--node_budget', type=int, default=1, help='node budget per node')
    parser.add_argument('--edge_budget', type=int, default=3, help='edge budget')
    parser.add_argument('--feature_budget', type=float, default=1, help='feature budget multiplier, dummy for continuous case')
    parser.add_argument('--accumulation_step', type=int, default=1, help='step to update params')
    parser.add_argument('--save_dir', type=str, default='result', help='save dir for injected nodes and edges')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor (default: 0.99)')


    args = parser.parse_args()
    device = torch.device('cuda', args.device)
    final_accs = []
    att = main(args)