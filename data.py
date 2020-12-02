import scipy.sparse as sp
import scipy.io as sio
import pandas as pd
import torch
import numpy as np
import networkx as nx
import csv
import sys
import pickle as pkl
from datetime import date, datetime
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components
# from Anomaly_Generation import anomaly_injection
from utils import partition_graph
from scipy.sparse import csc_matrix
from time import perf_counter
from torch.utils.data import DataLoader, Dataset
from networkx.readwrite import json_graph
import json


class DataLoader:
    def __init__(self, features, idx_l, idx_u, b_size):
        self.features = features
        # self.label = label
        self.idx_labeled = idx_l
        self.idx_unlabeled = idx_u
        self.bs = b_size

    def getBatch(self):
        # idx_x = []
        y_batch = []
        idx_x = np.random.choice(self.idx_labeled, size=int(self.bs / 2), replace=False).tolist()
        idx_x += np.random.choice(self.idx_unlabeled, size=int(self.bs / 2), replace=False).tolist()
        y_batch = np.concatenate((np.ones(int(self.bs / 2)), np.zeros(int(self.bs / 2))))
        # for i in range(self.bs):
        #     if i % 2 == 0:
        #         idx_x.append(np.random.choice(self.idx_labeled, size=1, replace=False))
        #         y_batch.append(1)
        #     else:
        #         idx_x.append(np.random.choice(self.idx_unlabeled, size=1, replace=False))
        #         y_batch.append(0)
        idx_x = np.array(idx_x).flatten()
        return self.features[idx_x], torch.FloatTensor(y_batch)

class DataLoaderN:
    def __init__(self, feature, l_list, ul_list, b_size, b_size_qry, nb_task, device):
        self.feature = feature
        self.labeled_l = l_list
        self.unlabeled_l = ul_list
        self.bs = b_size
        self.bs_qry = b_size_qry
        self.nb_task = nb_task
        self.device = device
    def getBatch(self, qry):
        # idx_l = []
        feature_l = []
        label_l = []
        feature_l_qry = []
        label_l_qry = []
        for i in range(self.nb_task):
            # idx_t = []
            # label_t = []
            # idx_t_qry = []
            # label_t_qry = []
            # print(type(self.labeled_l[i]))
            # input('...')
            idx_t = np.random.choice(self.labeled_l[i], size=int(self.bs / 2), replace=False).tolist()
            idx_t += np.random.choice(self.unlabeled_l[i], size=int(self.bs / 2), replace=False).tolist()
            label_t = np.concatenate((np.ones(int(self.bs / 2)), np.zeros(int(self.bs / 2))))
            # for j in range(self.bs):
            #     if j % 2 == 0:
            #         idx_t.append(np.random.choice(self.labeled_l[i], size=1, replace=False)[0])
            #         label_t.append(1)
            #     else:
            #         idx_t.append(np.random.choice(self.unlabeled_l[i], size=1, replace=False)[0])
            #         label_t.append(0)
            feature_l.append(self.feature[i][idx_t].to(self.device))
            label_l.append(torch.FloatTensor(label_t).to(self.device))
            if qry:
                idx_t_qry = np.random.choice(self.labeled_l[i], size=int(self.bs_qry / 2), replace=False).tolist()
                idx_t_qry += np.random.choice(self.unlabeled_l[i], size=int(self.bs_qry / 2), replace=False).tolist()
                label_t_qry = np.concatenate((np.ones(int(self.bs_qry / 2)), np.zeros(int(self.bs_qry / 2))))
                feature_l_qry.append(self.feature[i][idx_t_qry].to(self.device))
                label_l_qry.append(torch.FloatTensor(label_t_qry).to(self.device))

            # print(idx_t)
            # input('...')


        return feature_l, label_l, feature_l_qry, label_l_qry


def remove_values(arr1, arr2):

    res = [e for e in arr1 if e not in arr2]
    return np.array(res)


def load_yelp(file):
    data = sio.loadmat(file)
    network = data['Network'].astype(np.float)
    labels = data['Label'].flatten()
    attributes = data['Attributes'].astype(np.float)

    return network, attributes, labels

def load_data(d):
    # data = sio.loadmat("data/{}.mat".format(data_name))
    data = sio.loadmat(d)
    network = data['Network'].astype(np.float)
    labels = data['Label'].flatten()
    attributes = data['Attributes'].astype(np.float)

    return network, attributes, labels

def normalize_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def normalize_feature(feature):
    # Row-wise normalization of sparse feature matrix
    rowsum = np.array(feature.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(feature)
    return mx

def sp_matrix_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def SGC_process(data_name, degree, l_ratio, tr_ratio):
    adj, features, labels = load_data(data_name)
    adj = normalize_adjacency(adj)
    # split training and validation data
    idx_anomaly = np.nonzero(labels == 1)[0]
    idx_normal = np.nonzero(labels == 0)[0]
    np.random.shuffle(idx_anomaly)
    np.random.shuffle(idx_normal)
    [ano_train, ano_test] = np.array_split(idx_anomaly, [int(tr_ratio * len(idx_anomaly))])
    [nor_train, nor_test] = np.array_split(idx_normal, [int(tr_ratio * len(idx_normal))])
    idx_test = np.concatenate((ano_test, nor_test)).tolist()
    nb_ano = int(len(idx_anomaly) * l_ratio)
    nb_ano = 10
    idx_labeled = np.random.choice(ano_train, size=nb_ano, replace=False)
    idx_unlabeled = remove_values(idx_anomaly, idx_labeled)
    idx_unlabeled = np.concatenate((nor_train, idx_unlabeled)).tolist()

    adj = sp_matrix_to_torch_sparse_tensor(adj).float()
    features = normalize_feature(features)
    features = torch.FloatTensor(features.toarray())
    labels = torch.FloatTensor(labels)
    #compute S^K*X
    for i in range(degree):
        features = torch.spmm(adj, features)

    return features, labels, idx_labeled, idx_unlabeled, idx_test

# load_credit(0.8, 0.061)
# def load_credit(tr, lr):
#     file = "data/creditcard.csv"
#     idx_normal = []
#     features = []
#     labels = []
#     with open(file, newline='') as csvfile:
#         reader = csv.reader(csvfile, delimiter=',')
#         # print(type(reader))
#         for row in reader:
#             labels.append(float(row[-1]))
#             f = []
#             row_f = row[1:-1]
#             for v in row_f:
#                 f.append(float(v))
#             features.append(f)
#     features = np.array(features)
#     labels = np.array(labels)
#     print(features.shape)
#     # print(features[0])
#     n_samples = len(features)
#     print(len(labels))
#     features = normalize(features, axis=0, norm='l2')
#     idx_anomaly = np.nonzero(labels)[0]
#     idx_labeled_anomaly = np.random.choice(idx_anomaly, size=int(len(idx_anomaly) * lr), replace=False)
#     print(len(idx_labeled_anomaly))
#     idx_unlabeled = np.delete(np.arange(n_samples), idx_labeled_anomaly)
#     train_unlabeled = np.random.choice(idx_unlabeled, size=int(tr * n_samples) - len(idx_labeled_anomaly), replace=False)
#     idx_train = np.concatenate([idx_labeled_anomaly, train_unlabeled])
#     idx_test = np.delete(np.arange(n_samples), idx_train)
#     y_train = np.concatenate([labels[idx_labeled_anomaly], labels[train_unlabeled]])
#     y_test = labels[idx_test]
#     features = torch.FloatTensor(features)
#     labels = torch.LongTensor(labels)
#     y_train = torch.LongTensor(y_train)
#     y_test = torch.LongTensor(y_test)
#     print(features[0])

#     return features, labels, idx_train, y_train, idx_test, y_test, idx_labeled_anomaly, train_unlabeled

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def largest_connected_components(adj, n_components=1):
    _, component_indices = connected_components(adj)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
    print("Selecting {0} largest connected components".format(n_components))
    return nodes_to_keep

def load_pub_ori(dataset_str, nb_graphs):
    """
        Loads input data from gcn/data directory
        ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
        ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
        All objects above must be saved using python pickle module.
        :param dataset_str: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
    # data = {}
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/Pubmed/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/Pubmed/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).toarray()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    print(type(features))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).tocsc()
    print(type(adj))
    nb_nodes = adj.shape[0]
    label = np.zeros((nb_nodes, 1))
    step = 4000
    print(adj.shape)
    print(features.shape)
    pubmed_stats = 'data/Pubmed/pubmed_data_stats.txt'
    with open(pubmed_stats, 'a') as f:
        f.write('data_name,\tgraph_size,\t#anomalies\n')
        for t in range(4):
            nodes = np.random.permutation(nb_nodes)
            for i in range(nb_graphs):
                data = {}
                # partition the graph into several subgraphs
                nodes_t = nodes[i * step: (i + 1) * step]
                adj_s = adj[nodes_t][:, nodes_t]
                feature_s = features[nodes_t]
                label_s = label[nodes_t]
                node_s = largest_connected_components(adj_s)
                adj_s = adj_s[node_s][:, node_s]
                feature_s = feature_s[node_s]
                # print(type(feature_s))
                label_s = label_s[node_s]
                data["Network"] = adj_s
                data["Attributes"] = feature_s
                data["Label"] = label_s
                graph_size = adj_s.shape[0]
                ano_ratio = 0.0824
                divide = [0.8, 0.2]
                ano_a = int(graph_size * ano_ratio * divide[0])
                ano_s = int(graph_size * ano_ratio * divide[1])
                clique_size = ano_a // 10
                save_name = "data/Pubmed/Pubmed_{}_{}_{}.mat".format(str(t), str(i), date.today().strftime("%Y_%m_%d"))
                data_stat = "{},\t{},\t{}\n".format(save_name, str(graph_size), str(ano_a + ano_s))
                print("data stats: " + data_stat)
                f.write(data_stat)
                anomaly_injection(data, ano_s, ano_a, clique_size, 50, save_name)
    f.close()
    # return data #adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_pub(dataset_str, nb_graphs):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/Pubmed/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/Pubmed/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).toarray()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    # print(type(features))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph)).tocsc()
    # print(type(adj))
    nb_nodes = adj.shape[0]
    label = np.zeros((nb_nodes, 1))
    # print(adj.shape)
    # print(features.shape)
    pubmed_stats = 'data/Pubmed/pubmed_data_stats_{}.txt'.format(datetime.now().strftime("%m_%d"))
    #partition graph
    nb_runs = 5
    with open(pubmed_stats, 'a') as f:
        for i in range(nb_runs):
            node_list = partition_graph(adj, nb_graphs=5, nb_nodes=3800)
            ind = 0
            # input('...')
            for nodes in node_list:
                data = {}
                adj_s = adj[nodes][:, nodes]
                feature_s = features[nodes]
                label_s = label[nodes]
                node_s = largest_connected_components(adj_s)
                adj_s = adj_s[node_s][:, node_s]
                feature_s = feature_s[node_s]
                # print(type(feature_s))
                label_s = label_s[node_s]
                data["Network"] = adj_s
                data["Attributes"] = feature_s
                data["Label"] = label_s
                graph_size = adj_s.shape[0]
                ano_ratio = 0.0624
                divide = [0.5, 0.5]
                ano_a = int(graph_size * ano_ratio * divide[0])
                ano_s = int(graph_size * ano_ratio * divide[1])
                clique_size = 15
                save_name = "data/Pubmed/Pubmed_{}_{}_{}.mat".format(str(i), str(ind), datetime.now().strftime("%m_%d"))
                data_stat = "{},\t{},\t{}\n".format(save_name, str(graph_size), str(ano_a + ano_s))
                print("data stats: " + data_stat)
                f.write(data_stat)
                anomaly_injection(data, ano_s, ano_a, clique_size, 50, save_name)
                ind += 1
    f.close()

def load_reddit(nb_graphs):
    g_file = "data/reddit/reddit-G.json"
    feature_file = 'data/reddit/reddit-feats.npy'
    id_file = 'data/reddit/reddit-id_map.json'
    nodes = []
    with open(id_file) as j_file:
        idmap = json.load(j_file)
    for k in idmap:
        nodes.append(idmap[k])

    with open(g_file) as j_file:
        g_data = json.load(j_file)
    # edge_list = pd.DataFrame(columns=['source', 'target'])
    edge_list = []
    edge_data = g_data['links']
    for e in edge_data:
        edge_list.append((e['source'], e['target']))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edge_list)
    feature = np.load(feature_file)
    print(type(feature))
    # feature = csc_matrix(feature)
    nb_nodes = feature.shape[0]
    print(len(list(g.nodes)))
    print(feature.shape)
    adj = nx.adjacency_matrix(g)
    print(type(adj))
    adj = csc_matrix(adj)
    label = np.zeros((nb_nodes, 1))
    step = 18000
    reddit_stats = 'data/reddit/reddit_data_stats_' + date.today().strftime("%m_%d") + '.txt'
    with open(reddit_stats, 'a') as f:
        f.write('data_name,\tgraph_size,\tnb_edges,\t#anomalies\n')
        for t in range(1):
            nodes = np.random.permutation(nb_nodes)
            for i in range(nb_graphs):
                data = {}
                nodes_t = nodes[i * step: (i + 1) * step]
                adj_s = adj[nodes_t][:, nodes_t]
                feature_s = feature[nodes_t]
                label_s = label[nodes_t]
                node_s = largest_connected_components(adj_s)
                adj_s = adj_s[node_s][:, node_s]
                feature_s = feature_s[node_s]
                # print(type(feature_s))
                label_s = label_s[node_s]
                data["Network"] = adj_s
                data["Attributes"] = feature_s
                data["Label"] = label_s
                edge_count = np.count_nonzero(adj_s.toarray())
                graph_size = adj_s.shape[0]
                ano_ratio = 0.05
                divide = [0.5, 0.5]
                ano_a = int(graph_size * ano_ratio * divide[0])
                ano_s = int(graph_size * ano_ratio * divide[1])
                clique_size = 15
                save_name = "data/reddit/reddit_{}_{}.mat".format(str(i), date.today().strftime("%Y_%m_%d"))
                data_stat = "{},\t{},\t{},\t{}\n".format(save_name, str(graph_size), str(edge_count), str(ano_a + ano_s))
                print("data stats: " + data_stat)
                f.write(data_stat)
                anomaly_injection(data, ano_s, ano_a, clique_size, 50, save_name)
def main():
    pass

if __name__  == '__main__':
    # main()
    # load_pub("pubmed", nb_graphs=5)
    load_reddit(10)