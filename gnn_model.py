import torch
import torch_geometric
import torch.nn.functional as F

import torch.nn as nn

from torch_geometric.nn import (GCNConv, GATConv, SAGEConv, GraphConv, GINConv, GATv2Conv, 
                                ChebConv, SGConv, GINEConv, TAGConv, CuGraphSAGEConv)
__gnn_type_ls__ = ['gcn', 'gat', 'sage', 'gin', 'gatv2', 'tagconv', 'chevconv',
                 'sgconv', 'gine', 'graphconv']
def gnn_type_map(gnn_type, input_dim, hidden_dim):
    if gnn_type == 'gcn':
        return GCNConv(input_dim, hidden_dim)
    elif gnn_type == 'gat':
        return GATConv(input_dim, hidden_dim)
    elif gnn_type == 'sage':
        return SAGEConv(input_dim, hidden_dim)
    elif gnn_type == 'gin':
        return GINConv(nn.Linear(input_dim, hidden_dim))
    elif gnn_type == 'gatv2':
        return GATv2Conv(input_dim, hidden_dim)
    elif gnn_type == 'chevconv':
        return ChebConv(input_dim, hidden_dim, K=2)
    elif gnn_type == 'tagconv':
        return TAGConv(input_dim, hidden_dim)
    elif gnn_type == 'gine':
        return GINEConv(nn.Linear(input_dim, hidden_dim))
    elif gnn_type == 'sgconv':
        return SGConv(input_dim, hidden_dim, K=2)
    elif gnn_type == 'graphconv':
        return GraphConv(input_dim, hidden_dim)
    else:
        raise ValueError('Invalid GNN type')
    

class GraphLinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 drop_ratio = 0, gnn_type = 'gcn', aggr_type = 'sum',
                 **kwargs):
        super(GraphLinkPredictor, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type

        self.gnn_ls = nn.ModuleList()
        self.batch_norm_ls = nn.ModuleList()

        self.graph_encoder = GraphEncoder(input_dim, hidden_dim, num_layers, gnn_type)
        self.tf_linear = nn.Linear(hidden_dim[-1], output_dim)
        self.gene_linear = nn.Linear(hidden_dim[-1], output_dim)

        self.linear_decoder = nn.Linear(2 * output_dim, 1)

    def forward(self, x, edge_index, edge_index_label, edge_weight=None):
        h = self.encode(x, edge_index, edge_weight)
        h = torch.tanh(h)
        tf_embed = F.dropout(F.elu(self.tf_linear(h)), p=0.2)
        target_embed = F.dropout(F.elu(self.gene_linear(h)), p=0.2)

        tf_emb = tf_embed[edge_index_label[:, 0], :]
        gene_emb = target_embed[edge_index_label[:, 1], :]

        score = self.decode(tf_emb, gene_emb)
        return h, tf_emb, gene_emb, score

    def encode(self, x, edge_index, edge_weight=None):
        return self.graph_encoder(x, edge_index)

    def decode(self, tf_embed, gene_embed):
        concat_emb = torch.cat([tf_embed, gene_embed], dim = 1)
        prob = self.linear_decoder(concat_emb)
        return prob
    
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, gnn_types = 'gcn',
                 **kwargs):
        super(GraphEncoder, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.gnn_types = gnn_types
        self.graph_encoder = nn.ModuleList()
        self.graph_encoder.append(GCNConv(input_dim, hidden_dims[0], aggr='mean'))
        for i in range(1, num_layers):
            self.graph_encoder.append(GCNConv(hidden_dims[i-1], hidden_dims[i], aggr='mean'))

    def forward(self, x, edge_index):
        h = torch.tanh(self.graph_encoder[0](x, edge_index))
        for i in range(1, self.num_layers):
            if i == self.num_layers - 1:
                h = self.graph_encoder[i](h, edge_index)
            else:
                h = torch.tanh(self.graph_encoder[i](h, edge_index))
        return h
    
class GraphContrasiveModel(nn.Module):
    def __init__(self,encoder, aug1, aug2):
        super(GraphContrasiveModel, self).__init__()

        self.encoder = encoder
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self, x, edge_index, train_data):
    

        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)

        embed1, tf_embed1, target_embed1, pred1 = self.encoder(x, edge_index1, train_data)
        embed2, tf_embed2, target_embed2, pred2 = self.encoder(x, edge_index2, train_data)
            
        return embed1, tf_embed1, target_embed1, pred1, embed2, tf_embed2, target_embed2, pred2


