from collections import defaultdict
import torch
import numpy as np
from torch import nn


class PositionAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(PositionAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class MeanPositionAggregator(PositionAggregator):
  def __init__(self, device):
    super(MeanPositionAggregator, self).__init__(device)


  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        message_dim = messages[node_id][0][0].shape[0]
        position_dim = (message_dim - 1) // 2
        to_update_node_ids.append(node_id)
        neighbors = torch.stack([m[0][:position_dim] for m in messages[node_id]])
        node_self = messages[node_id][0][0][position_dim:2*position_dim].unsqueeze(0)
        all_neighbors = torch.cat((neighbors, node_self), dim=0)
        unique_messages.append(torch.mean(all_neighbors, dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class ExponentialPositionAggregator(PositionAggregator):
    def __init__(self, device, alpha, beta):
        super(ExponentialPositionAggregator, self).__init__(device)
        self.alpha = alpha
        self.beta = beta

    def aggregate(self, node_ids, messages):
        """
        Aggregate messages using a custom exponential decay formula based on the timestamp.

        :param node_ids: A list of node ids of length batch_size
        :param messages: A tensor of shape [batch_size, message_length]
        :param timestamps: A tensor of shape [batch_size]
        :return: Tuple of unique node ids, aggregated messages, and their corresponding timestamps
        """
        unique_node_ids = np.unique(node_ids)
        unique_messages = []
        unique_timestamps = []

        to_update_node_ids = []
        n_messages = 0

        for node_id in unique_node_ids:
          if len(messages[node_id]) > 0:
            n_messages += len(messages[node_id])
            message_dim = messages[node_id][0][0].shape[0]
            position_dim = (message_dim - 1) // 2
            to_update_node_ids.append(node_id)
            neighbors = torch.stack([m[0][:position_dim] * (self.alpha ** (
              -torch.relu(self.beta * (m[0][2*position_dim:])))) for m in messages[node_id]])
            node_self = messages[node_id][0][0][position_dim:2*position_dim].unsqueeze(0)
            all_neighbors = torch.cat((neighbors, node_self), dim=0)
            unique_messages.append(torch.mean(all_neighbors, dim=0))
            unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


def get_position_aggregator(aggregator_type, device, alpha=2, beta=1.0):
  if aggregator_type == "mean":
    return MeanPositionAggregator(device=device)
  elif aggregator_type == "exp":
    return ExponentialPositionAggregator(device=device, alpha=alpha, beta=beta)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
