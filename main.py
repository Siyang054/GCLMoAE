import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from gnn_model import GraphLinkPredictor, GraphContrasiveModel
from moe_model import GNN_MoE_LinkPredictor

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
# from PytorchTools import EarlyStopping
from sklearn.preprocessing import StandardScaler

import os, sys
import argparse
import random
from utils import *

import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast


import pandas as pd
import numpy as np
import random
import os
import sys
import time
import argparse
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation
import GCL.losses as L
import GCL.augmentors as A
from GCL.models import DualBranchContrast


parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('-epochs', type=int, default=20, help='Number of epoch.')
parser.add_argument('-num_head', type=list, default=[3, 3], help='Number of head attentions.')
parser.add_argument('-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('-hidden_dims', type=int, default=[64,64], help='The dimension of hidden layer')
parser.add_argument('-output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--num_layers', type = int, default=2)
parser.add_argument('--drop_ratio', type=float, default=0)
parser.add_argument('-batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('-loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('-seed', type=int, default=8, help='Random seed')
parser.add_argument('-Type', type=str, default='dot', help='score metric')
parser.add_argument('-flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('-reduction', type=str, default='concate', help='how to integrate multihead attention')
parser.add_argument('-sample', type=str, default='sample1', help='sample')
# parser.add_argument('-cell_type', type=str, default='hHEP', help='cell_type')
parser.add_argument('--dataset_name', type = str, default = 'hHEP')
parser.add_argument('--dataset_path', type=str, default='Dataset/Benchmark Dataset/', help='dataset save path')
parser.add_argument('--net_type', type=str, default='Specific', help='network type')
parser.add_argument('--TF_nums', type=int, default=1000, help='network scale')
parser.add_argument('--num_experts', type = int, default=8)
parser.add_argument('--topk', type=int, default=8)

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def train(model, contrast_model, optimizer, scheduler, train_load, data_feature, adj):

    running_loss = 0.0
    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)

        embed1, _, _, pred1, embed2, _, _, pred2 = model(data_feature, adj, train_x)

        con_loss = contrast_model(h1=embed1, h2=embed2)

        if args.flag:
            pred1 = torch.softmax(pred1, dim=1)
            pred2 = torch.softmax(pred2, dim=1)
        else:
            pred1 = torch.sigmoid(pred1)
            pred2 = torch.sigmoid(pred2)

        loss_BCE1 = F.binary_cross_entropy(pred1, train_y)
        loss_BCE2 = F.binary_cross_entropy(pred2, train_y)

        loss = loss_BCE1 + loss_BCE2 + 0.5 * con_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        
    return running_loss

# Load Data
def main(args):
    (data_feature, train_load, train_data, validation_data, test_data,
        train_edge_index, test_edge_index, val_edge_index) = load_rcRNAdataset(args.dataset_path, args.net_type,
                                                                                args.dataset_name,
                                                                            args.TF_nums, device, args.seed, args.sample)

    print('feature shape ', data_feature.shape)
    adj = train_edge_index

    # Construct Model
    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='L2L', intraview_negs=False).to(device)
    encoder = GNN_MoE_LinkPredictor(data_feature.shape[1], args.hidden_dims, args.output_dim,
                                    args.num_layers, args.num_experts, args.topk).to(device)
    print(encoder)
    data_feauture = data_feature.to(device)

    adj = adj.to(device)
    train_data = train_data.to(device)
    test_data = test_data.to(device)
    validation_data = validation_data.to(device)

    model_path = 'model'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Data Augmentation
    aug1 = A.Identity()
    aug2 = A.EdgeRemoving(pe=0.2)

    model = GraphContrasiveModel(encoder=encoder, aug1=aug1, aug2=aug2)
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    # Train model
    AUC_Threshold = 0
    auc_test, auprc_test = 0, 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = train(model, contrast_model, optimizer, scheduler, train_load, data_feature, adj)

        model.eval()
        _, _, _, score1, _, _, _, score2 = model(data_feature, adj, validation_data)
        if args.flag:
            score = torch.softmax(score1, dim=1)
        else:
            score = torch.sigmoid(score1)
        
        AUC, AUPR, AUPR_norm = Evaluation(
            y_pred=score, y_true=validation_data[:, -1], flag=args.flag)
        
        print('Epoch:{}'.format(epoch + 1),
            'train loss:{:.5F}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))
        
        if AUC > AUC_Threshold:
            AUC_Threshold = AUC
            model.eval()

            _, _, _, score1, _, _, _, score2 = model(data_feature, adj, test_data)

            if args.flag:
                score = torch.softmax(score1, dim=1)
            else:
                score = torch.sigmoid(score1)
            
            auc_test, auprc_test, auprc_norm_test = Evaluation(
                y_pred=score, y_true=test_data[:, -1], flag=args.flag)

    # Load best model and test

    print('test_AUC:{:.3F}'.format(auc_test), 'test_AUPR:{:.3F}'.format(auprc_test))
    return auc_test, auprc_test


if __name__ == '__main__':
    auc_test, auprc_test = main(args)
    print(f'Test auc:{auc_test:.4f}, test auprc:{auprc_test:.4f}')
