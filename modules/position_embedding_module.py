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


class PositionAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder,
               time_encoder, n_layers, n_node_features, n_edge_features,
               n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True,
               position_dim=8, position_embedding_dim=12):
      super(PositionAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension, device,
                                                      n_heads, dropout,
                                                      use_memory)

      self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(n_node_features + position_embedding_dim,
                                                           embedding_dimension) for _ in range(n_layers)])
      self.linear_11 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension, embedding_dimension)
                                            for _ in range(n_layers)])
      self.linear_2 = torch.nn.ModuleList([torch.nn.Linear(n_node_features + position_embedding_dim,
                                                           embedding_dimension) for _ in range(n_layers)])
      self.linear_22 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension, embedding_dimension)
                                            for _ in range(n_layers)])

      self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
                                                  n_node_features=embedding_dimension,
                                                  n_neighbors_features=embedding_dimension,
                                                  n_edge_features=n_edge_features,
                                                  time_dim=n_time_features,
                                                  n_head=n_heads,
                                                  dropout=dropout,
                                                  output_dimension=n_node_features)
                                                  for _ in range(n_layers)])


      # Integrated position embedding functionality
      self.position_decoder = nn.Sequential(
        nn.Linear(position_dim, position_dim * 2),
        nn.ReLU(),
        nn.Linear(position_dim * 2, position_embedding_dim),
        # nn.ReLU(),
        # nn.Linear(position_embedding_dim // 2, position_embedding_dim)
      )

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, position_memory,
                        n_neighbors=20, time_diffs=None, use_time_proj=True):

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(
      torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]
    source_position_features = self.position_decoder(position_memory[source_nodes, :])
    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features


    source_node_features = torch.cat([source_node_features,
                                      source_position_features], dim=1)
    if n_layers == 0:
      return source_node_features
    else:
      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers-1,
                                                           n_neighbors=n_neighbors,
                                                           position_memory=position_memory)

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
                                                   n_neighbors=n_neighbors,
                                                   position_memory=position_memory)

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
                                        mask,
                                        edge_deltas_torch)

      return source_embedding

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask,
                timestamps):
    # source node features decoder
    source_position_decoding = source_node_features[:, self.embedding_dimension:]
    source_node_features = self.linear_1[n_layer - 1](source_node_features)
    source_node_features = torch.relu(source_node_features)
    source_node_features = self.linear_11[n_layer - 1](source_node_features)

    # neighbor node features decoder
    reshaped_neighbors_embedding = self.linear_2[n_layer - 1](neighbor_embeddings.view(-1, neighbor_embeddings.shape[-1]))
    reshaped_neighbors_embedding = torch.relu(reshaped_neighbors_embedding)
    reshaped_neighbors_embedding = self.linear_22[n_layer - 1](reshaped_neighbors_embedding)
    neighbor_embeddings = reshaped_neighbors_embedding.view(neighbor_embeddings.shape[0], neighbor_embeddings.shape[1], -1)

    # aggregate node features
    attention_model = self.attention_models[n_layer - 1]
    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    source_embedding = torch.cat([source_embedding, source_position_decoding], dim=1)
    return source_embedding


def get_position_embedding_module(module_type, node_features, edge_features,
                         memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features,
                         n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True,
                         position_dim=4, position_embedding_dim=12):

  if module_type == "position_attn":
    module = PositionAttentionEmbedding(node_features=node_features,
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
                                        use_memory=use_memory,
                                        position_dim=position_dim,
                                        position_embedding_dim=position_embedding_dim)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))
  return module
