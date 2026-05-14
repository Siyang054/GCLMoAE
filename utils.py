import joblib
import os
import torch
import numpy as np
import pandas as pd
import random
import json, logging, sys
import scipy.sparse as sp

import math
import logging.config 
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,average_precision_score


def init_seed(seed=2020):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def save_emb(score_emb, save_path):

    if len(score_emb) == 6:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x1, x2= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x1,
        'node_emb_with_valid_edges': x2

        }
        
    elif len(score_emb) == 5:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, x= score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        'node_emb': x
        }
    elif len(score_emb) == 4:
        pos_valid_pred,neg_valid_pred, pos_test_pred, neg_test_pred, = score_emb
        state = {
        'pos_valid_score': pos_valid_pred,
        'neg_valid_score': neg_valid_pred,
        'pos_test_score': pos_test_pred,
        'neg_test_score': neg_test_pred,
        }
   
    torch.save(state, save_path)

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            best_results = []

            for r in self.results:
                r = 100 * torch.tensor(r)
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')

            r = best_result[:, 0].float()
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')

            r = best_result[:, 1].float()
            best_valid_mean = round(r.mean().item(), 2)
            best_valid_var = round(r.std().item(), 2)

            best_valid = str(best_valid_mean) +' ' + '±' +  ' ' + str(best_valid_var)
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 2].float()
            best_train_mean = round(r.mean().item(), 2)
            best_train_var = round(r.std().item(), 2)
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')


            r = best_result[:, 3].float()
            best_test_mean = round(r.mean().item(), 2)
            best_test_var = round(r.std().item(), 2)
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

            mean_list = [best_train_mean, best_valid_mean, best_test_mean]
            var_list = [best_train_var, best_valid_var, best_test_var]


            return best_valid, best_valid_mean, mean_list, var_list

def get_logger(name, log_dir, config_dir):
	
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


class load_data():
    def __init__(self, data, normalize=True):
        self.data = data
        self.normalize = normalize

    def data_normalize(self,data):
        standard = StandardScaler()
        epr = standard.fit_transform(data.T)

        return epr.T


    def exp_data(self):
        data_feature = self.data.values

        if self.normalize:
            data_feature = self.data_normalize(data_feature)

        data_feature = data_feature.astype(np.float32)

        return data_feature
    
# def Evaluation(y_true, y_pred,flag=False):
#     if flag:
#         # y_p = torch.argmax(y_pred,dim=1)
#         y_p = y_pred[:,-1]
#         y_p = y_p.cpu().detach().numpy()
#         y_p = y_p.flatten()
#     else:
#         y_p = y_pred.cpu().detach().numpy()
#         y_p = y_p.flatten()


#     y_t = y_true.cpu().numpy().flatten().astype(int)

#     AUC = roc_auc_score(y_true=y_t, y_score=y_p)


#     AUPR = average_precision_score(y_true=y_t,y_score=y_p)
#     AUPR_norm = AUPR/np.mean(y_t)


#     return AUC, AUPR, AUPR_norm


def Evaluation(y_true, y_pred,flag=False):
    y_p = y_pred.cpu().detach().numpy()
    # print(y_p.shape)
    # print(y_true.shape)

    y_t = y_true.cpu().numpy()
    # y_true_pos = y_true[y_true==1]
    # y_true_neg = y_true[y_true==0]
    # y_t = np.concatenate([y_true_pos,y_true_neg])
    #y_t = y_true.cpu().numpy().flatten().astype(int)

    AUC = roc_auc_score(y_true=y_t, y_score=y_p)


    AUPR = average_precision_score(y_true=y_t,y_score=y_p)
    AUPR_norm = AUPR/np.mean(y_t)


    return AUC, AUPR, AUPR_norm


class scRNADataset(Dataset):
    def __init__(self,train_set,num_gene,flag=False):
        super(scRNADataset, self).__init__()
        self.train_set = train_set
        self.num_gene = num_gene
        self.flag = flag


    def __getitem__(self, idx):
        train_data = self.train_set[:,:2]
        train_label = self.train_set[:,-1]

        # if self.flag:
        #     train_len = len(train_label)
        #     train_tan = np.zeros([train_len,2])
        #     train_tan[:,0] = 1 - train_label
        #     train_tan[:,1] = train_label
        #     train_label = train_tan

        data = train_data[idx].astype(np.int64)
        label = train_label[idx].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.train_set)

def construct_edge_index(tf_gene_pairs, tf, is_undirected=True):
    edge_index = []
    tf_ls = tf.tolist()
    for src, dst, label in tf_gene_pairs:
        if is_undirected:
            if label == 1:
                edge_index.append([src, dst])
                # if dst in tf_ls:
                edge_index.append([dst, src])
        else:
            if label == 1:
                edge_index.append([src, dst])
                if dst in tf_ls:
                    edge_index.append([dst, src])
    edge_index = np.array(edge_index, dtype = np.int64)
    return edge_index

def load_rcRNAdataset(datasetpath, net_type, dataset_name, TF_nums, device, seed = None, sample = 'sample1'):
    exp_file = (datasetpath + net_type + ' Dataset/' + dataset_name + '/TFs+' +
                str(TF_nums) + '/BL--ExpressionData.csv')
    tf_file = datasetpath + net_type + ' Dataset/' + dataset_name + '/TFs+' + str(TF_nums) + '/TF.csv'
    # target_file = '../' + datasetpath + net_type + ' Dataset/' + dataset_name + '/TFs+' + str(TF_nums) + '/Target.csv'

    # if seed is None:
    # print(os.listdir(os.getcwd() + '/Data/' + net_type + '/'))
    # train_file = os.getcwd() + '/proc_data/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) +  '/Train_set.csv'
    # val_file = os.getcwd() + '/proc_data/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/Validation_set.csv'
    # test_file = os.getcwd() + '/proc_data/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/Test_set.csv'
    train_file = os.getcwd() + '/Data/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/' + sample +  '/Train_set.csv'
    val_file = os.getcwd() + '/Data/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/' + sample +  '/Validation_set.csv'
    test_file = os.getcwd() + '/Data/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/' + sample + '/Test_set.csv'

    # else:
    #     train_file = os.getcwd() + '/1_1_8_ratio/Train_Val_Test_split/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/' + str(seed) + '/Train_set.csv'
    #     val_file = os.getcwd() + '/1_1_8_ratio/Train_Val_Test_split/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/' + str(seed) + '/Validation_set.csv'
    #     test_file = os.getcwd() + '/1_1_8_ratio/Train_Val_Test_split/' + net_type + '/' + dataset_name + ' ' + str(TF_nums) + '/' + str(seed) + '/Test_set.csv'

    data_input = pd.read_csv(exp_file, index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    # feature = torch.from_numpy(feature)
    tf = torch.from_numpy(pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int64))
    # target = pd.read_csv(target_file, index_col=0)['index'].values.astype(np.int64)

    train_data = pd.read_csv(train_file, index_col=0).values
    validation_data = pd.read_csv(val_file, index_col=0).values
    test_data = pd.read_csv(test_file, index_col=0).values

    # train_data = np.concatenate([train_data, validation_data], axis=0)

    train_load = scRNADataset(train_data, feature.shape[0], flag=False)

    train_edge_index = construct_edge_index(train_data, tf)
    test_edge_index = construct_edge_index(test_data, tf)
    val_edge_index = construct_edge_index(validation_data, tf)
    # train_edge_index = np.concatenate((train_edge_index, val_edge_index), axis=0)
    # test_edge_index = np.concatenate((train_edge_index, val_edge_index), axis=0)
    sorted_indices_train_edge = np.lexsort((train_edge_index[:, 1], train_edge_index[:, 0]))
    train_edge_index = train_edge_index[sorted_indices_train_edge]
    # sorted_indices_test_edge = np.lexsort((test_edge_index[:, 1], test_edge_index[:, 0]))
    # test_edge_index = test_edge_index[sorted_indices_test_edge]
    # sorted_indices_val_edge = np.lexsort((test_edge_index[:, 1], test_edge_index[:, 0]))
    # val_edge_index = test_edge_index[sorted_indices_val_edge]
    test_edge_index = train_edge_index
    val_edge_index = train_edge_index

    data_feature = torch.from_numpy(feature).to(device)
    train_data = torch.from_numpy(train_data).to(device)
    validation_data = torch.from_numpy(validation_data).to(device)
    test_data  = torch.from_numpy(test_data).to(device)

    train_edge_index = torch.tensor(train_edge_index, dtype=torch.long).T.to(device)
    val_edge_index = torch.tensor(val_edge_index, dtype=torch.long).T.to(device)
    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).T.to(device)

    return (data_feature, train_load, train_data, validation_data, test_data,
            train_edge_index, test_edge_index, val_edge_index)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self,save_dir, patience=7,verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                           
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                           
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.save_dir = save_dir
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            #self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
       
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')     
        torch.save(model.state_dict(), self.save_dir+'.pkl')
        self.val_loss_min = val_loss

def construct_2hop_graph(num_nodes, edge_index):
    num_edges = edge_index.shape[1]
    hash_table = {}
    two_hop_edge_index = []
    for j in range(num_edges):
        start_node_idx = edge_index[0,j].item()
        if start_node_idx in hash_table:
            hash_table[start_node_idx].append(j)
        else:
            hash_table[start_node_idx] = [j]
    for node_idx in range(num_nodes):
        if node_idx not in hash_table: # this is weird but some graphs has isolated nodes.
            continue
        for first_edge_idx in hash_table[node_idx]:
            first_edge = edge_index[:,first_edge_idx]
            hop_node_idx = first_edge[1].item()
            for second_edge_idx in hash_table[hop_node_idx]:
                second_edge = edge_index[:,second_edge_idx]
                if second_edge[1].item() == first_edge[0].item(): 
                    continue # we don't consider 1->2 and 2->1 as 1--two-hop-->1
                if second_edge[1].item() in hash_table[first_edge[0].item()]: # note: first_edge[0].item() == node_idx
                    continue # we don't consider 1->2 as a two-hop path if there is a one-hop path between 1 & 2.
                two_hop_edge = [first_edge[0].item(), second_edge[1].item()]
                two_hop_edge_index.append(two_hop_edge)
    
    two_hop_edge_index = torch.Tensor(two_hop_edge_index).T.long()
    return two_hop_edge_index


def Network_Statistic(data_type,net_scale,net_type):

    if net_type =='STRING':
        dic = {'hESC500': 0.024, 'hESC1000': 0.021, 'hHEP500': 0.028, 'hHEP1000': 0.024, 'mDC500': 0.038,
               'mDC1000': 0.032, 'mESC500': 0.024, 'mESC1000': 0.021, 'mHSC-E500': 0.029, 'mHSC-E1000': 0.027,
               'mHSC-GM500': 0.040, 'mHSC-GM1000': 0.037, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.045}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale



    elif net_type == 'Non-Specific':

        dic = {'hESC500': 0.016, 'hESC1000': 0.014, 'hHEP500': 0.015, 'hHEP1000': 0.013, 'mDC500': 0.019,
               'mDC1000': 0.016, 'mESC500': 0.015, 'mESC1000': 0.013, 'mHSC-E500': 0.022, 'mHSC-E1000': 0.020,
               'mHSC-GM500': 0.030, 'mHSC-GM1000': 0.029, 'mHSC-L500': 0.048, 'mHSC-L1000': 0.043}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Specific':
        dic = {'hESC500': 0.164, 'hESC1000': 0.165,'hHEP500': 0.379, 'hHEP1000': 0.377,'mDC500': 0.085,
               'mDC1000': 0.082,'mESC500': 0.345, 'mESC1000': 0.347,'mHSC-E500': 0.578, 'mHSC-E1000': 0.566,
               'mHSC-GM500': 0.543, 'mHSC-GM1000': 0.565,'mHSC-L500': 0.525, 'mHSC-L1000': 0.507}

        query = data_type + str(net_scale)
        scale = dic[query]
        return scale

    elif net_type == 'Lofgof':
        dic = {'mESC500': 0.158, 'mESC1000': 0.154}

        query = 'mESC' + str(net_scale)
        scale = dic[query]
        return scale

    else:
        raise ValueError

class G_vocab():
    def __init__(self, node, type, index):
        super().__init__()
        self.node = node
        self.index = index
        self.type = type

def get_node2vocab(type_gene_dict): 
    gene_type_dict = {}
    for type, _ in type_gene_dict.items():
        for node in type_gene_dict[type]:
            gene_type_dict[node] = type
    node2vocab = {}
    for gene_node, type in gene_type_dict.items():
        node2vocab[gene_node] = G_vocab(node=gene_node, type=gene_type_dict[gene_node],
                                        index=gene_node)
    return node2vocab

def get_type_genes_dict(data_feature, TF_list_file): 
    type_gene_dict = {}
    nodes_num = np.arange(len(data_feature))
    All_TF_list = pd.read_csv(TF_list_file, index_col=0)["index"].values 
    Target_only = np.setdiff1d(nodes_num, All_TF_list)
    assert len(nodes_num) == len(Target_only) + len(All_TF_list)
    type_gene_dict[0] = All_TF_list  # 0: Regulator gene
    type_gene_dict[1] = Target_only  # 1:Target gene
    return type_gene_dict, nodes_num

def get_train_exp_adj_pairs(type_gene_dict, training_nodes_pairs_index, net_work_nodes_num,
                            directed=True):
    # each gene type class
    all_nodes_vocab = get_node2vocab(type_gene_dict)
    # training_nodes_pairs_index = pd.read_csv(train_set_file, index_col=0).values

    # decouple graph
    self_loop = np.tile(np.arange(net_work_nodes_num), (2, 1)).T
    TF_TF_row, TF_TF_col = [], []  # 0-0
    TF_Target_row, TF_Target_col = [], []  # 0-1
        # prior directed GRN based on train_sets
    training_original_edges = np.array([edge[:2] for edge in training_nodes_pairs_index if edge[2] == 1], dtype=np.int32)  # the link in train set as prior graph
        # decouple graph
    for edge in training_original_edges:
        if all_nodes_vocab[edge[0]].type == 0 and all_nodes_vocab[edge[1]].type == 0:  # 0-0
            TF_TF_row.append(all_nodes_vocab[edge[0]].index)
            TF_TF_col.append(all_nodes_vocab[edge[1]].index)

        if all_nodes_vocab[edge[0]].type == 0 and all_nodes_vocab[edge[1]].type == 1:  # 0-1
            TF_Target_row.append(all_nodes_vocab[edge[0]].index)
            TF_Target_col.append(all_nodes_vocab[edge[1]].index)
    TF_TF_index = np.vstack((np.array(TF_TF_row), np.array(TF_TF_col)))  # TF_TF directed graph
    TF_Target_index = np.vstack((np.array(TF_Target_row), np.array(TF_Target_col))) # TF_Target directed graph

    if directed == True:
        TF_TF_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_TF_index.shape[1]), (TF_TF_index[0, :], TF_TF_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_TF_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_TF_sqarse_numpy.row, TF_TF_sqarse_numpy.col]),
            torch.FloatTensor(TF_TF_sqarse_numpy.data),
            torch.Size(TF_TF_sqarse_numpy.shape))

        TF_Target_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_Target_index.shape[1]), (TF_Target_index[0, :], TF_Target_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_Target_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_Target_sqarse_numpy.row, TF_Target_sqarse_numpy.col]),
            torch.FloatTensor(TF_Target_sqarse_numpy.data),
            torch.Size(TF_Target_sqarse_numpy.shape))
        training_adj_edges = [TF_TF_sqarse_tensor, TF_Target_sqarse_tensor]
        return training_original_edges, training_adj_edges, training_nodes_pairs_index
    else:
        # Prior GRN graph adj for explicit embedding
        training_undirected_edge = np.concatenate((np.stack((training_original_edges[:, 1],
                                                             training_original_edges[:, 0]), axis=1),
                                                   training_original_edges), axis=0)  #
        training_undirected_edge = np.unique(np.concatenate((training_undirected_edge,
                                                             self_loop), axis=0), axis=0)
        # sub-graph
        TF_TF_Transpose_index = TF_TF_index[[1, 0], :]  # The reverse graph of TF_TF directed graph
        TF_Target_Transpose_index = TF_Target_index[[1, 0], :]  #  The reverse graph of TF_Target directed graph
        # adj based on index
        TF_TF_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_TF_index.shape[1]), (TF_TF_index[0, :], TF_TF_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_TF_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_TF_sqarse_numpy.row, TF_TF_sqarse_numpy.col]),
            torch.FloatTensor(TF_TF_sqarse_numpy.data),
            torch.Size(TF_TF_sqarse_numpy.shape))

        TF_Target_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_Target_index.shape[1]), (TF_Target_index[0, :], TF_Target_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_Target_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_Target_sqarse_numpy.row, TF_Target_sqarse_numpy.col]),
            torch.FloatTensor(TF_Target_sqarse_numpy.data),
            torch.Size(TF_Target_sqarse_numpy.shape))

        TF_TF_Transpose_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_TF_Transpose_index.shape[1]), (TF_TF_Transpose_index[0, :], TF_TF_Transpose_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_TF_Transpose_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_TF_Transpose_sqarse_numpy.row, TF_TF_Transpose_sqarse_numpy.col]),
            torch.FloatTensor(TF_TF_Transpose_sqarse_numpy.data),
            torch.Size(TF_TF_Transpose_sqarse_numpy.shape))

        TF_Target_Transpose_sqarse_numpy = sp.coo_matrix(
            (np.ones(TF_Target_Transpose_index.shape[1]),
             (TF_Target_Transpose_index[0, :], TF_Target_Transpose_index[1, :])),
            shape=(net_work_nodes_num, net_work_nodes_num))
        TF_Target_Transpose_sqarse_tensor = torch.sparse.FloatTensor(
            torch.LongTensor([TF_Target_Transpose_sqarse_numpy.row, TF_Target_Transpose_sqarse_numpy.col]),
            torch.FloatTensor(TF_Target_Transpose_sqarse_numpy.data),
            torch.Size(TF_Target_Transpose_sqarse_numpy.shape))
        training_adj_edges = [TF_TF_sqarse_tensor, TF_Target_sqarse_tensor, TF_TF_Transpose_sqarse_tensor,
                              TF_Target_Transpose_sqarse_tensor]
        # training_adj_edges = [TF_Target_sqarse_tensor]
        return training_undirected_edge, training_adj_edges, training_nodes_pairs_index
    
def adj2saprse_tensor(adj):
    coo = adj.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.from_numpy(values).float()

    adj_sp_tensor = torch.sparse_coo_tensor(i, v, coo.shape)
    return adj_sp_tensor
