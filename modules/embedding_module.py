import torch
from torch import nn, Tensor
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                        n_neighbors=20, time_diffs=None, use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where
      we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider
      in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:
      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers-1,
                                                           n_neighbors=n_neighbors)

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


class PositionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder,
               time_encoder, n_layers, n_node_features, n_edge_features,
               n_time_features, embedding_dimension, device, num_nodes: int,
               n_heads=2, dropout=0.1, use_memory=True,
               node_embedding_module=None):
      super(PositionEmbedding, self).__init__(node_features, edge_features, memory,
                                              neighbor_finder, time_encoder, n_layers,
                                              n_node_features, n_edge_features,
                                              n_time_features,
                                              embedding_dimension, device,
                                              n_heads, dropout,
                                              use_memory)

      # Integrated position embedding functionality
      self.position_embedding = nn.Embedding(num_nodes, embedding_dimension)
      if node_embedding_module is not None:
        self.node_embedding_module = node_embedding_module

  def position_message(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor) -> Tensor:
      pos_msg = self.position_embedding(raw_msg.long()).reshape(
          z_src.shape[0], -1) + z_src  # Ensure dimensions match for addition
      return torch.cat([z_src, z_dst, pos_msg, t_enc], dim=-1)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                      n_neighbors=20, time_diffs=None, use_time_proj=True):

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:
      source_node_conv_embeddings = self.compute_embedding(memory,
                                                            source_nodes,
                                                            timestamps,
                                                            n_layers=n_layers-1,
                                                            n_neighbors=n_neighbors)

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                    neighbors,
                                                    np.repeat(timestamps, n_neighbors),
                                                    n_layers=n_layers - 1,
                                                    n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
      assert (n_layers >= 0)

      source_nodes_torch = torch.from_numpy(np.asarray(source_nodes)).long().to(self.device)
      timestamps_torch = torch.unsqueeze(torch.from_numpy(np.asarray(timestamps)).float().to(self.device), dim=1)

      # Retrieve embeddings from memory
      source_node_features = memory[source_nodes_torch]
      source_nodes_time_embedding = self.time_encoder(torch.zeros_like(timestamps_torch))

      # Apply integrated position message
      pos_messages = self.position_message(source_node_features, source_node_features, source_nodes_torch, source_nodes_time_embedding)

      if n_layers == 0:
          return pos_messages  # Return position messages directly if no further recursive layer

      # Recursive embedding for neighbors
      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(source_nodes, timestamps, n_neighbors=n_neighbors)
      neighbors_torch = torch.from_numpy(np.asarray(neighbors)).long().to(self.device)
      edge_idxs = torch.from_numpy(np.asarray(edge_idxs)).long().to(self.device)
      edge_deltas = timestamps[:, np.newaxis] - edge_times
      edge_deltas_torch = torch.from_numpy(np.asarray(edge_deltas)).float().to(self.device)

      neighbor_embeddings = self.compute_embedding(memory, neighbors.flatten(), np.repeat(timestamps, n_neighbors),
                                                    n_layers=n_layers - 1, n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)
      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      # Combine embeddings from current node and its neighbors
      combined_embeddings = self.aggregate(n_layers, pos_messages, source_nodes_time_embedding,
                                            neighbor_embeddings, edge_time_embeddings, edge_features, mask)

      return combined_embeddings

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask,
                timestamps, V=None, R=None, src_idx=None, neigh_idx=None,
                roots=None, targets=None, node_level=False):
    neighbors_features = torch.cat([neighbor_embeddings, edge_features],
                                   dim=2)
    mask = ~mask
    timestamps = timestamps.unsqueeze(-1)
    number_neighbors = torch.sum(mask, dim=1).unsqueeze(-1).unsqueeze(-1)

    R = R / (R.sum(dim=2, keepdim=True) + 1e-04)

    neighbor_embeddings = self.linear_1[n_layer - 1](torch.cat(
      [neighbors_features,
       R[neigh_idx.flatten(), roots, :].view((neigh_idx.shape[0], neigh_idx.shape[1], R.shape[2])),
       R[neigh_idx.flatten(), targets, :].view((neigh_idx.shape[0], neigh_idx.shape[1], R.shape[2]))],
      dim=2))

    neighbor_embeddings = torch.relu(neighbor_embeddings)
    neighbor_embeddings = self.linear_11[n_layer - 1](neighbor_embeddings)

    neighbors_sum = neighbor_embeddings * (self.alpha ** (
      -torch.relu(self.beta * (timestamps)) / torch.sqrt(number_neighbors + 1e-4)))

    neighbors_sum = torch.sum(neighbors_sum * mask.unsqueeze(-1), dim=1)

    source_embedding = torch.cat([neighbors_sum, source_node_features, R[src_idx, roots[0::neigh_idx.shape[1]], :],
                                    R[src_idx, targets[0::neigh_idx.shape[1]], :]], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)
    source_embedding = torch.relu(source_embedding)
    source_embedding = self.linear_22[n_layer - 1](source_embedding)

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features,
                         memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features,
                         n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True, position_eocoding=False):

  if module_type == "graph_attention":
    module = GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout,
                                    use_memory=use_memory)
  elif module_type == "graph_sum":
    module = GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    module = IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    module = TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  elif module_type == "exp":
    module = None
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))

  if position_eocoding:
    module = PositionEmbedding(node_features=node_features,
                               edge_features=edge_features,
                               memory=memory,
                               neighbor_finder=neighbor_finder,
                               time_encoder=time_encoder,
                               n_layers=n_layers,
                               n_node_features=n_node_features,
                               n_edge_features=n_edge_features,
                               n_time_features=n_time_features,
                               embedding_dimension=embedding_dimension,
                               device=device,
                               num_nodes=node_features.shape[0],
                               n_heads=n_heads, dropout=dropout,
                               use_memory=use_memory,
                               node_embedding_module=module)

  return module
