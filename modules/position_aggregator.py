from collections import defaultdict
import torch
import numpy as np
from torch import nn


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
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


class LastMeanPositionMessageAggregator(MessageAggregator):
  def __init__(self, device, position_dim):
    super(LastMeanPositionMessageAggregator, self).__init__(device)
    self.position_dim = position_dim
    self.position_message_dim = 2 * position_dim + 1

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            node_message = messages[node_id][-1][0]
            position_encoding = torch.mean(torch.stack([m[0][-self.position_message_dim:-(self.position_dim+1)]
                                                        +m[0][-(self.position_dim+1):-1]
                                                        for m in messages[node_id]]), dim=0)
            unique_messages.append(torch.cat((node_message, position_encoding)))
            unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMeanPositionMessageAggregator(MessageAggregator):
  def __init__(self, device, position_dim):
    super(LastMeanPositionMessageAggregator, self).__init__(device)
    self.position_dim = position_dim
    self.position_message_dim = 2 * position_dim + 1

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(
               torch.mean(torch.stack([torch.cat((m[0][:self.position_message_dim],
                                                  m[0][-self.position_message_dim:-(self.position_dim+1)]
                                                  +m[0][-(self.position_dim+1):-1]), dim=0)
                                                  for m in messages[node_id]]), dim=0))
            unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class LastExponentialPositionMessageAggregator(MessageAggregator):
  def __init__(self, device, position_dim, alpha, beta):
    super(LastExponentialPositionMessageAggregator, self).__init__(device)
    self.position_dim = position_dim
    self.position_message_dim = 2 * position_dim + 1
    self.alpha = alpha
    self.beta = beta

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []

    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            node_message = messages[node_id][-1][0]
            position_encoding = torch.mean(torch.stack([self.alpha * m[0][-self.position_message_dim:-(self.position_dim+1)]
                                                        *(torch.exp(-torch.relu(self.beta * m[0][-1])))
                                                        +m[0][-(self.position_dim+1):-1]
                                                        for m in messages[node_id]]), dim=0)

            unique_messages.append(torch.cat((node_message, position_encoding)))
            unique_timestamps.append(messages[node_id][-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanExponentialPositionMessageAggregator(MessageAggregator):
    def __init__(self, device, position_dim, alpha, beta):
        super(MeanExponentialPositionMessageAggregator, self).__init__(device)
        self.position_dim = position_dim
        self.position_message_dim = 2 * position_dim + 1
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

        for node_id in unique_node_ids:
          if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(
               torch.mean(torch.stack([torch.cat((m[0][:self.position_message_dim],
                                                  self.alpha * m[0][-self.position_message_dim:-(self.position_dim+1)] * \
                                                    (torch.exp(-torch.relu(self.beta * m[0][-1]))) + \
                                                      m[0][-(self.position_dim+1):-1]), dim=0)
                                                  for m in messages[node_id]]), dim=0))
            unique_timestamps.append(messages[node_id][-1][1])

        unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
        unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

        return to_update_node_ids, unique_messages, unique_timestamps


def get_position_message_aggregator(message_aggregator_type, device, position_aggregator_type,
                                    position_dim, alpha=2, beta=1.0):
  if message_aggregator_type == "mean" and position_aggregator_type == "exp":
    return MeanExponentialPositionMessageAggregator(device=device,
                                         position_dim=position_dim,
                                         alpha=alpha,
                                         beta=beta)
  elif message_aggregator_type == "last" and position_aggregator_type == "exp":
    return LastExponentialPositionMessageAggregator(device=device,
                                         position_dim=position_dim,
                                         alpha=alpha,
                                         beta=beta)
  elif message_aggregator_type == "mean" and position_aggregator_type == "mean":
    return MeanMeanPositionMessageAggregator(device=device, position_dim=position_dim)
  elif message_aggregator_type == "last" and position_aggregator_type == "mean":
    return LastMeanPositionMessageAggregator(device=device, position_dim=position_dim)
  else:
    raise ValueError(f"Message aggregator {message_aggregator_type} not recognized.")
