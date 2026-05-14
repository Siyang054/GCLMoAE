import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv

__aggr_ls__ = ['mean', 'mean', 'sum', 'sum', 'max', 'max', 'att', 'att']

def experts_ls(input_dim, hidden_dim, aggr_type):
    if aggr_type in ['mean', 'sum', 'max']:
        return GCNConv(input_dim, hidden_dim, aggr = aggr_type)
    else:
        return GATConv(input_dim, hidden_dim)

def gnn_type_map(gnn_type, input_dim, hidden_dim):
    if gnn_type == 'gcn':
        return GCNConv(input_dim, hidden_dim)
    elif gnn_type == 'gat':
        return GATConv(input_dim, hidden_dim)
    elif gnn_type == 'sage':
        return SAGEConv(input_dim, hidden_dim)
    elif gnn_type == 'gin':
        return GINConv(nn.Linear(input_dim, hidden_dim))
    else:
        raise ValueError('Invalid GNN type')

class MV_MOE(torch.nn.Module):
    def __init__(self, num_views, input_dim, hidden_dim, output_dim,
                 num_layers, num_experts, k, **kwargs):
        super(MV_MOE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_views = num_views
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.k = k
        self.gate_models = nn.ModuleList()
        assert num_experts == num_views

        self.mv_moe_ls = nn.ModuleList()
        for layer in range(num_layers):
            moe_model_ls = nn.ModuleList()
            for i in range(num_views):
                if layer == 0:
                    moe_model_ls.append(GCNConv(input_dim, hidden_dim))
                else:
                    moe_model_ls.append(GCNConv(hidden_dim, hidden_dim))
            self.mv_moe_ls.append(moe_model_ls)
            if layer == 0:
                self.gate_models.append(nn.Linear(input_dim, num_experts))
            else:
                self.gate_models.append(nn.Linear(hidden_dim, num_experts))
        
        self.tf_linear = nn.Linear(hidden_dim, output_dim)
        self.target_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index_ls, edge_index_label):
        h_ls = [x,]
        for layer in range(self.num_layers):
            feature = h_ls[-1]
            expert_outputs_ls = []
            for i in range(self.num_views):
                expert_output = self.mv_moe_ls[layer][i](feature, edge_index_ls[i])
                expert_outputs_ls.append(expert_output)
            
            clean_logits = self.gate_models[layer](feature)

            logits = clean_logits
            top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
            top_k_logits = top_logits[:, :self.k]
            top_k_indices = top_indices[:, :self.k]
            top_k_gates = F.softmax(top_k_logits, dim = 1)

            zeros = torch.zeros_like(logits, requires_grad=True)
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
            gate_weights = F.softmax(gates, dim = 1).unsqueeze(dim=-1)
            expert_outputs_fuse = (torch.stack(expert_outputs_ls, dim=1)*gate_weights).sum(dim=1)
            h_ls.append(expert_outputs_fuse)
        h = h_ls[-1]

        tf_emb = F.tanh(self.tf_linear(h))
        target_emb = F.tanh(self.target_linear(h))
        pred = self.decode(tf_emb, target_emb, edge_index_label)
        return pred
    
    def decode(self, tf_emb, target_emb, edge_index):
        prob = torch.mul(tf_emb[edge_index[:, 0], :], target_emb[edge_index[:, 1], :])
        prob = torch.sum(prob, dim = 1).view(-1, 1)
        return prob


class GNN_MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, k, 
                 use_res=False, **kwargs):
        super(GNN_MoE, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k 
        self.use_res = use_res
        
        self.gnn_model_ls = nn.ModuleList() 
        for i in range(num_experts):
            self.gnn_model_ls.append(experts_ls(input_dim, hidden_dim, __aggr_ls__[i]))
        self.gates = nn.Linear(input_dim, num_experts)
        self.res_proj = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        expert_outputs_ls = []
        for i in range(self.num_experts):
            expert_outputs = F.tanh(self.gnn_model_ls[i](x, edge_index))
            expert_outputs_ls.append(expert_outputs)
        expert_outputs = torch.stack(expert_outputs_ls, dim=1)

        clean_logits = self.gates(x)

        logits = clean_logits
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = F.softmax(top_k_logits, dim = 1)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        gate_weights = F.softmax(gates, dim = 1).unsqueeze(dim=-1)

        # gate_weights = F.softmax(self.gates(x), dim = 1).unsqueeze(dim=-1)
        if self.use_res:
            output = torch.sum(expert_outputs * gate_weights, dim=1) + self.res_proj(x)
        else:
            output = torch.sum(expert_outputs * gate_weights, dim=1)
        return output

class GNN_MoE_LinkPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_layers, num_experts, k,
                 gnn_type = 'gcn', **kwargs):
        super(GNN_MoE_LinkPredictor, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.k = k
        self.output_dim = output_dim

        self.gnn_moe_ls = nn.ModuleList()
        self.gnn_moe_ls.append(GNN_MoE(input_dim, hidden_dims[0], num_experts, k))
        for i in range(1, num_layers):
            self.gnn_moe_ls.append(GNN_MoE(hidden_dims[i-1], hidden_dims[i], 
                                           num_experts, k))

        self.tf_linear = nn.Linear(hidden_dims[-1], output_dim)
        self.target_linear = nn.Linear(hidden_dims[-1], output_dim)

        self.linear_decoder = nn.Linear(2 * output_dim, 1)

    def forward(self, x, edge_index, edge_label_index):
        h = x
        for i in range(self.num_layers):
            h = self.gnn_moe_ls[i](h, edge_index)
        emb = F.tanh(h)
        tf_emb = F.elu(self.tf_linear(emb))
        target_emb = F.elu(self.target_linear(emb))
        
        tf_emb = tf_emb[edge_label_index[:, 0], :]
        target_emb = target_emb[edge_label_index[:, 1], :]
        score = self.decode(tf_emb, target_emb)

        return emb, tf_emb, target_emb, score

    def decode(self, tf_embed, gene_embed):
        concat_emb = torch.cat([tf_embed, gene_embed], dim = 1)
        prob = self.linear_decoder(concat_emb)
        return prob
        
