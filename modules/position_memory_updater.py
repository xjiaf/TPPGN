from torch import nn
import torch


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class PositionMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(PositionMemoryUpdater, self).__init__()
    self.memory = memory
    # self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.memory_dimension = memory_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    # updated_memory = self.memory_updater(unique_messages, memory)
    updated_node_memory = self.memory_updater(unique_messages[:, :self.message_dimension], memory[:, :self.memory_dimension])
    updated_memory = torch.cat((updated_node_memory, unique_messages[:, self.message_dimension:]), dim=1)

    self.memory.set_memory(unique_node_ids, updated_memory)

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past" \

    updated_memory = self.memory.memory.data.clone()
    # updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])
    updated_node_memory = self.memory_updater(unique_messages[:, :self.message_dimension],
                                              updated_memory[unique_node_ids][:, :self.memory_dimension])
    updated_memory[unique_node_ids] = torch.cat((
      updated_node_memory, unique_messages[:, self.message_dimension:]), dim=1)

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUPositionMemoryUpdater(PositionMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUPositionMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNPositionMemoryUpdater(PositionMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNPositionMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_position_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUPositionMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNPositionMemoryUpdater(memory, message_dimension, memory_dimension, device)
