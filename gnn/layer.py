import torch
from torch import nn
from torch.nn import functional as F
from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("layer.PNALayer")
class PNALayer(layers.MessagePassingBase, core.Configurable):

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(PNALayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * 2 * input_dim)
        else:
            self.relation = nn.Embedding(num_relation * 2, input_dim)

        # test sparse conv
        adjacency = torch.rand(1, 1, 1, device=self.device).to_sparse()
        relation_input = torch.rand(1, 32, device=self.device)
        input = torch.rand(1, 32, device=self.device)
        functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul="mul")

    def message_and_aggregate(self, graph, input):
        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)
        node_in, node_out, relation = graph.edge_list.t()

        degree_out = getattr(graph, "pna_degree_out", graph.degree_out)
        degree_out = degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation * 2, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        if isinstance(graph, data.PackedGraph):
            relation = relation + graph.num_relation * graph.edge2graph
            relation_input = relation_input.flatten(0, 1)
            adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out, relation]), graph.edge_weight,
                                                (graph.num_node, graph.num_node, batch_size * graph.num_relation))
        else:
            relation_input = relation_input.transpose(0, 1).flatten(1)
            adjacency = graph.adjacency
        adjacency = adjacency.transpose(0, 1)

        relation_input = relation_input.type(input.dtype)

        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add")
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add")
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max")
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add")
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add")
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max")
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min")
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=1e-6).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            degree_mean = getattr(graph, "pna_degree_mean", scale.mean())
            scale = scale / degree_mean
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)
        if not isinstance(graph, data.PackedGraph):
            update = update.view(len(update), batch_size, -1)
        return update

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
