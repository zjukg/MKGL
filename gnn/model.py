import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import  scatter_add

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .util import VirtualTensor, Range, RepeatGraph
from .util import bincount, variadic_topks
from .layer import *

@R.register("PNA")
class PNA(nn.Module, core.Configurable):

    def __init__(self, base_layer, num_layer, num_mlp_layer=2, remove_one_hop=False):
        super(PNA, self).__init__()

        self.num_relation = base_layer.num_relation
        self.remove_one_hop = remove_one_hop
        self.layers = nn.ModuleList()
        for i in range(num_layer):
            self.layers.append(core.Configurable.load_config_dict(base_layer.config_dict()))
        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [1])

    def aggregate(self, graph, input_embeds):
        layer_input = input_embeds
        for layer in self.layers:
            hidden = layer(graph, layer_input)
            if self.short_cut:
                hidden = hidden + layer_input
            layer_input = hidden
            
        return hidden

    def init_input_embeds(self, graph, input_embeds, input_index):
        input_embeds = torch.zeros(graph.num_node, input_embeds.shape[-1], device=input_embeds.device)
        input_embeds[input_index] = input_embeds
        return input_embeds


    def forward(self, graph, input_embeds, input_index):
        graph = graph.undirected(add_inverse=True)
        input_embeds = self.init_input_embeds(input_embeds, input_index)
        output = self.aggregate(graph, input_embeds)
        return output



@R.register("ConditionedPNA")
class ConditionedPNA(PNA, core.Configurable):

    def __init__(self, base_layer, num_layer, num_mlp_layer=2, node_ratio=0.1, degree_ratio=1, test_node_ratio=None, test_degree_ratio=None,
                 break_tie=False, **kwargs):
        
        super().__init__(base_layer, num_layer, num_mlp_layer=num_mlp_layer, **kwargs)

        self.node_ratio = node_ratio
        self.degree_ratio = degree_ratio
        self.test_node_ratio = test_node_ratio or node_ratio
        self.test_degree_ratio = test_degree_ratio or degree_ratio
        self.break_tie = break_tie

        feature_dim = base_layer.output_dim + base_layer.input_dim
        self.rel_embedding = nn.Embedding(base_layer.num_relation * 2, base_layer.input_dim)
        self.linear = nn.Linear(feature_dim, base_layer.output_dim)
        self.mlp = layers.MLP(base_layer.output_dim, [feature_dim] * (num_mlp_layer - 1) + [1])


    def forward(self, h_index, r_index, t_index, hidden_states, rel_hidden_states, graph, score_text_embs, all_index):
        if self.training:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)
        graph = graph.undirected(add_inverse=True)
        h_index, t_index, r_index = self.negative_sample_to_tail(
            h_index, t_index, r_index)
        
        batch_size = len(h_index)
        graph = RepeatGraph(graph, batch_size)
        offset = graph.num_cum_nodes - graph.num_nodes
        h_index = h_index + offset.unsqueeze(-1).to(h_index.device)
        t_index = t_index + offset.unsqueeze(-1).to(t_index.device)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        rel_embeds = self.rel_embedding(r_index[:, 0]) 
        rel_embeds = rel_embeds.type(hidden_states.dtype) #+ rel_hidden_states

        input_embeds, init_score = self.init_input_embeds(graph, hidden_states, h_index[:, 0], score_text_embs, all_index, rel_embeds)
        score = self.aggregate(graph, h_index[:, 0], r_index[:, 0], input_embeds, rel_embeds, init_score)
        score = score[t_index]
        return score

    def aggregate(self, graph, h_index, r_index, input_embeds, rel_embeds, init_score):
        query = rel_embeds
        boundary, score = input_embeds, init_score
        hidden = boundary.clone()
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary
            graph.hidden = hidden
            graph.score = score
            graph.node_id = Range(graph.num_node, device=h_index.device)
            graph.pna_degree_out = graph.degree_out
        with graph.edge():
            graph.edge_id = Range(graph.num_edge, device=h_index.device)
        pna_degree_mean = (graph[0].degree_out + 1).log().mean()

        for layer in self.layers:
            edge_index = self.select_edges(graph, graph.score)
            subgraph = graph.edge_mask(edge_index, compact=True)
            subgraph.pna_degree_mean = pna_degree_mean

            layer_input = F.sigmoid(subgraph.score).unsqueeze(-1) * subgraph.hidden
            hidden = layer(subgraph, layer_input.type(torch.float32))
            out_mask = subgraph.degree_out > 0
            node_out = subgraph.node_id[out_mask]

            graph.hidden[node_out] = (graph.hidden[node_out] + hidden[out_mask]).type(graph.hidden[node_out].dtype)

            index = graph.node2graph[node_out]
            graph.score[node_out] = self.score(graph.hidden[node_out], query[index]).type(graph.score[node_out].dtype)

            data_dict, meta_dict = subgraph.data_by_meta("graph")
            graph.meta_dict.update(meta_dict)
            graph.__dict__.update(data_dict)

        return graph.score

    def init_input_embeds(self, graph, head_embeds, head_index, tail_embeds, tail_index,  rel_embeds):
        input_embeds = VirtualTensor.zeros(graph.num_node, rel_embeds.shape[1], device=rel_embeds.device, dtype=rel_embeds.dtype)
        
        
        input_embeds[tail_index] = tail_embeds.type(head_embeds.dtype)
        input_embeds[head_index] = head_embeds

        score = VirtualTensor.gather(self.score(torch.zeros_like(rel_embeds), rel_embeds), graph.node2graph) # zero all
        score[head_index] = self.score(head_embeds, rel_embeds)
            
        return input_embeds, score

    def score(self, hidden, rel_embeds):
        heuristic = self.linear(torch.cat([hidden, rel_embeds], dim=-1))
        x = hidden * heuristic
        score = self.mlp(x).squeeze(-1)
        return score


    def select_edges(self, graph, score):
        node_ratio = self.node_ratio if self.training else self.test_node_ratio
        degree_ratio = self.degree_ratio if self.training else self.test_degree_ratio
        ks = (node_ratio * graph.num_nodes).long()
        es = (degree_ratio * ks * graph.num_edges / graph.num_nodes).long()

        node_in = score.keys
        num_nodes = bincount(graph.node2graph[node_in], minlength=len(graph))
        ks = torch.min(ks, num_nodes)
        score_in = score[node_in]
        index = variadic_topks(score_in, num_nodes, ks=ks, break_tie=self.break_tie)[1]
        node_in = node_in[index]
        num_nodes = ks

        num_neighbors = graph.num_neighbors(node_in)
        num_edges = scatter_add(num_neighbors, graph.node2graph[node_in], dim_size=len(graph))
        es = torch.min(es, num_edges)

        num_edge_mean = num_edges.float().mean().clamp(min=1)
        chunk_size = max(int(1e7 / num_edge_mean), 1)
        num_nodes = num_nodes.split(chunk_size)
        num_edges = num_edges.split(chunk_size)
        es = es.split(chunk_size)
        num_chunk_nodes = [num_node.sum() for num_node in num_nodes]
        node_ins = node_in.split(num_chunk_nodes)

        edge_indexes = []
        for node_in, num_node, num_edge, e in zip(node_ins, num_nodes, num_edges, es):
            edge_index, node_out = graph.neighbors(node_in)
            score_edge = score[node_out]
            index = variadic_topks(score_edge, num_edge, ks=e, break_tie=self.break_tie)[1]
            edge_index = edge_index[index]
            edge_indexes.append(edge_index)
        edge_index = torch.cat(edge_indexes)

        return edge_index
    
    def remove_easy_edges(self, graph, h_index, t_index, r_index):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            any = -torch.ones_like(h_index_ext)
            pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
        else:
            pattern = torch.stack([h_index, t_index, r_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index