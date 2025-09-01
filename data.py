import numpy as np
import scipy.sparse as sp
import dgl
import random
import torch
from sklearn.model_selection import StratifiedKFold


class load_data(object):

    def __init__(self, dataset, device):
        if dataset == 'citeseer':
            self.g = dgl.data.CiteseerGraphDataset(verbose=False)[0]
            self.discrete_feat = True
        elif dataset == 'cora':
            self.g = dgl.data.CoraGraphDataset(verbose=False)[0]
            self.discrete_feat = True
        elif dataset == 'pubmed':
            self.g = dgl.data.PubmedGraphDataset(verbose=False)[0]
            self.discrete_feat = False
        elif dataset == 'wiki_cs':
            self.g = dgl.data.WikiCSDataset(verbose=False)[0]
            self.discrete_feat = False
        elif dataset == 'co_computer':
            self.g = dgl.data.AmazonCoBuyComputerDataset(verbose=False)[0]
            self.discrete_feat = True
        elif dataset == 'co_photo':
            self.g = dgl.data.AmazonCoBuyPhotoDataset(verbose=False)[0]
            self.discrete_feat = True
        elif dataset == 'reddit':
            self.g = load_npz('reddit')
            self.discrete_feat = False
        elif dataset == 'ogbproducts':
            self.g = load_npz('ogbproducts')
            self.discrete_feat = False
        else:
            raise Exception('Dataset not implemented Error.')


        if 'train_mask' not in self.g.ndata:
            train_mask, val_mask, test_mask = cross_validation_gen(self.g.ndata['label'])
            self.g.ndata['train_mask'], self.g.ndata['val_mask'], self.g.ndata['test_mask'] = \
                train_mask[:,0], val_mask[:,0], test_mask[:,0]
        elif dataset == 'wiki_cs':
            self.g.ndata['train_mask'], self.g.ndata['val_mask'], self.g.ndata['test_mask'] = \
                self.g.ndata['train_mask'][:,0].bool(), self.g.ndata['val_mask'][:,0].bool(), self.g.ndata['test_mask'].bool()


        idx_train = torch.where(self.g.ndata['train_mask'])[0]
        idx_val = torch.where(self.g.ndata['val_mask'])[0]
        idx_test = torch.where(self.g.ndata['test_mask'])[0]
        self.index_split = {'train_index': idx_train, 'val_index': idx_val, 'test_index':idx_test}

        # self.g.ndata['feat'] = F.normalize(self.g.ndata['feat'])
        self.feature_budget = (self.g.ndata['feat'] > 0).float().sum(1).mean() if self.discrete_feat else self.g.ndata['feat'].sum(1).mean().to(device)
        self.mu = self.g.ndata['feat'].mean(0).to(device)
        self.sigma = self.g.ndata['feat'].std(0).to(device)
        self.g = dgl.add_self_loop(self.g).to(device)


def load_npz(dataset):
    if dataset == 'reddit':
        dataset = '12k_reddit'
    elif dataset == 'ogbproducts':
        dataset = '10k_ogbproducts'
    else:
        raise Exception('Not implemented err.')

    file_name = 'datasets/{}.npz'.format(dataset)
    """Load a SparseGraph from a Numpy binary file.
    Parameters
    ----------
    file_name : str
        Name of the file to load.
    Returns
    -------
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.
    """
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                    loader['adj_indptr']), shape=loader['adj_shape'])

        if 'attr_data' in loader:
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                         loader['attr_indptr']),shape=loader['attr_shape'])
        else:
            attr_matrix = None

        labels = loader.get('labels')
    g = dgl.DGLGraph(adj_matrix)
    g.ndata['feat'] = torch.tensor(attr_matrix.todense()).float()
    split = np.load('datasets/{}_split.npy'.format(dataset),allow_pickle=True).item()
    train_mask = split['train']
    val_mask = split['val']
    test_mask = split['test']
    temp = torch.zeros(g.number_of_nodes(),dtype=torch.bool)
    temp[train_mask] = True
    g.ndata['train_mask'] = temp
    temp = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    temp[val_mask] = True
    g.ndata['val_mask'] = temp
    temp = torch.zeros(g.number_of_nodes(), dtype=torch.bool)
    temp[test_mask] = True
    g.ndata['test_mask'] = temp
    g.ndata['label'] = torch.tensor(labels)

    return g


def cross_validation_gen(y,k_flod=5):
    skf = StratifiedKFold(n_splits=k_flod)
    train_splits = []
    val_splits = []
    test_splits = []

    for larger_group, smaller_group in skf.split(y, y):
        train_y = y[smaller_group]
        sub_skf = StratifiedKFold(n_splits=2)
        train_split, val_split = next(iter(sub_skf.split(train_y,train_y)))
        train = torch.zeros_like(y, dtype=torch.bool)
        train[smaller_group[train_split]] = True
        val = torch.zeros_like(y, dtype=torch.bool)
        val[smaller_group[val_split]] = True
        test = torch.zeros_like(y, dtype=torch.bool)
        test[larger_group] = True
        train_splits.append(train.unsqueeze(1))
        val_splits.append(val.unsqueeze(1))
        test_splits.append(test.unsqueeze(1))

    return torch.cat(train_splits, dim=1), torch.cat(val_splits, dim=1), torch.cat(test_splits, dim=1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


