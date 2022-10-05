from this import d
import torch
from torch import nn
import torch.nn.functional as F

import networkx as nx

# In the weight generating function I should concatenate all the inputs and hidden states
# into a single column vector, then I should concatenate all the column vectors into a matrix
# Then I can multiply this matrix by Q, K, and V to get the queries, keys, and values.
# Once I have these three large matrices, I can use each node's in_edge list to compute the
# relevancies.
# directed_graph is an adjacency matrix converted to a tensorbool
# This would be more efficient if, instead of a directed graph, I provided a
# list of lists of predecessor nodes of each node.
# TODO:
# 1. Should the input, interneurons, and output neurons all have different update functions?
# can they all share the same weight generating function?


class DynamicAttentionNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_state_size,
        hidden_layer_size,
        adjacency_matrix,
        device="cpu",
    ):
        super(DynamicAttentionNetwork, self).__init__()

        self.adjacency_matrix = adjacency_matrix
        self.num_neurons = self.adjacency_matrix.shape[0]

        self.initial_hidden_states = nn.Parameter(
            torch.zeros(self.num_neurons, hidden_state_size)
        )

        self.step_size = nn.Parameter(torch.tensor(0.01))

        neuron_state_size = input_size + hidden_state_size
        self.state_update = StateUpdateFunction(
            neuron_state_size, hidden_layer_size, hidden_state_size
        )
        # self.state_update = nn.Linear(neuron_state_size, hidden_state_size, bias=False)
        self.predecessor_state_accumulator = AccumulatorFunction(
            neuron_state_size, self.adjacency_matrix, device
        )

        self.init_params()

    # this function initializes the weights of the model
    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # The input_states and hidden_states contain all input states and hidden states
    # for all the neurons in the graph and so are of size (input_size, num_nodes) and
    # (hidden_state_size, num_nodes), respectively.
    def forward(
        self,
        input_states,
        hidden_states=None,
        initial_output_mask=None,
        initial_outputs=None,
    ):
        if hidden_states is None:
            if initial_outputs is None:
                hidden_states = self.initial_hidden_states
            else:
                hidden_states = self.initial_hidden_states
                hidden_states = hidden_states * initial_output_mask + initial_outputs

        # make batches work
        if input_states.dim() == 2:
            if hidden_states.dim() == 2:
                hidden_states = torch.unsqueeze(hidden_states, 0)
                hidden_states = hidden_states.expand(input_states.shape[0], -1, -1)

            input_states = torch.unsqueeze(input_states, 2)

        neuron_states = torch.cat((input_states, hidden_states), 2)
        # print(neuron_states[0, :, 0])

        accumulated_predecessor_states = self.predecessor_state_accumulator(
            neuron_states
        )

        updated_hidden_states = hidden_states + self.step_size * self.state_update(
            accumulated_predecessor_states
        )
        return updated_hidden_states


class AccumulatorFunction(nn.Module):
    def __init__(self, neuron_state_size, adjacency_matrix, device):
        super(AccumulatorFunction, self).__init__()

        self.adjacency_mask = adjacency_matrix.transpose(0, 1)

        self.relevancies = torch.zeros(adjacency_matrix.shape, device=device)

        # Make the zero values -100000 so that softmax brings them to 0
        self.softmax_mask = -100000 * (
            torch.ones(self.adjacency_mask.shape, device=device) - self.adjacency_mask
        )

        self.Query = nn.Linear(neuron_state_size, neuron_state_size, bias=False)
        self.Key = nn.Linear(neuron_state_size, neuron_state_size, bias=False)
        self.Value = nn.Linear(neuron_state_size, neuron_state_size, bias=False)

    def forward(self, neuron_states):
        # can concat QKV to do one big matrix multiply, then separate
        queries = self.Query(neuron_states)
        keys = self.Key(neuron_states)
        values = self.Value(neuron_states)

        relevancies = torch.matmul(queries, keys.transpose(1, 2))
        relevancies = F.softmax(self.softmax_mask + relevancies, dim=2)

        self.relevancies = self.adjacency_mask * relevancies

        accumulated_predecessor_states = torch.matmul(
            self.adjacency_mask * relevancies, values
        )
        # print(accumulated_predecessor_states[0, :, 0])

        return accumulated_predecessor_states


# This is an MLP with a single hidden layer
class StateUpdateFunction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StateUpdateFunction, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = self.linear2(x)
        return x
