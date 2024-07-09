import torch
from torch import nn


class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None


class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()
    self.raw_message_dimension = raw_message_dimension
    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages):
    raw_node_message = raw_messages[:, :self.raw_message_dimension]
    node_messages = self.mlp(raw_node_message)
    messages = torch.cat((node_messages, raw_messages[:, self.raw_message_dimension:]), dim=1)
    return messages


class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):
    return raw_messages


def get_position_message_function(module_type, raw_message_dimension, message_dimension):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "identity":
    return IdentityMessageFunction()
