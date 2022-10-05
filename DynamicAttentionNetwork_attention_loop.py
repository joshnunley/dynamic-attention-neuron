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
# 2. I should look into different initialization methods (especially what NCA does) playing
#    with the model makes me think this is a big issue
# 3. I should make sure everything is on the GPU
# 4. I should look at the learning rate scheduling and normalization in the NCA model
# 5. I should look at making it so I can use batch training, which does not work currently
# 6. I should look into multi-headed attention

class DynamicAttentionNetwork(nn.Module):
    def __init__(self, input_size, hidden_state_size, adjacency_matrix, device):
        super(DynamicAttentionNetwork, self).__init__()

        self.device = device

        self.adjacency_matrix = adjacency_matrix
        self.num_neurons = self.directed_graph.shape[0]

        self.initial_hidden_states = nn.Parameter(torch.zeros(self.num_neurons, hidden_state_size))

        self.step_size = nn.Parameter(torch.zeros(1))

        neuron_state_size = input_size + hidden_state_size
        self.state_update = StateUpdateFunction(2*neuron_state_size, 128, hidden_state_size)
        self.predecessor_state_accumulator = AccumulatorFunction(neuron_state_size)
        
        self.init_params()
        
    # this function initializes the weights of the model
    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    # The input_states and hidden_states contain all input states and hidden states
    # for all the neurons in the graph and so are of size (input_size, num_nodes) and
    # (hidden_state_size, num_nodes), respectively.
    def forward(self, input_states, hidden_states=None):
        if hidden_states is None:
            hidden_states = self.initial_hidden_states
        
        neuron_states = torch.cat((input_states, hidden_states), 1)

        accumulated_predecessor_states = self.predecessor_state_accumulator(neuron_states, self.adjacency_matrix) 

        neuron_predecessor_states = torch.cat((neuron_states, accumulated_predecessor_states), 1)
        updated_hidden_states = hidden_states + self.step_size * self.state_update(neuron_predecessor_states)
        return updated_hidden_states

class AccumulatorFunction(nn.Module):
    def __init__(self, neuron_state_size, adjacency_matrix):
        super(AccumulatorFunction, self).__init__()

        self.adjacency_matrix = adjacency_matrix        

        self.Query = nn.Linear(neuron_state_size, neuron_state_size, bias=False)
        self.Key = nn.Linear(neuron_state_size, neuron_state_size, bias=False)
        self.Value = nn.Linear(neuron_state_size, neuron_state_size, bias=False)

    def forward(self, neuron_states):
        queries = self.Query(neuron_states)
        keys = self.Key(neuron_states)
        values = self.Value(neuron_states)

        accumulated_predecessor_states = torch.zeros(neuron_states.shape, device=self.device)
        # Can I speed this up by switching to sparse matrix multiplication
        # and doing everything at once? Maybe if I'm smart, my particular matrix 
        # is exploitably sparse.
        for neuron in range(self.num_neurons):
            # the predecessors of the neuron are in a row of the directed_graph
            # which is defined by a tensorbool adjacency matrix. Select
            # the row and repeat it so it can be used as a mask to select the keys.
            predecessors_mask = self.adjacency_matrix[:, neuron].reshape(-1, 1)

            # the keys for all of the neighbors are collected into a matrix
            # and multiplied by the query vector to compute the relevancies
            predecessor_keys = torch.masked_select(keys, predecessors_mask).view(-1, keys.shape[1])
            relevancies = torch.matmul(predecessor_keys, queries[neuron])
            relevancies = F.softmax(relevancies, dim=0)

            # the relevancies are multiplied by the values of the neighbors
            # and then summed to get the accumulated predecessor states
            predecessor_values = torch.masked_select(values, predecessors_mask).view(-1, values.shape[1])
            accumulated_predecessor_states[neuron] = torch.matmul(relevancies, predecessor_values)
        
        return accumulated_predecessor_states

# This is an MLP with a single hidden layer
class StateUpdateFunction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StateUpdateFunction, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = self.linear2(x)
        return x



# This is a simple test to make sure the weight generating neural network is working
# correctly. It should be able to learn the identity function.
if __name__ == "__main__":
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from([0, 1, 2, 3, 4])
    directed_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 0), (2, 1), (2, 3), (3, 2), (2, 4)])

    weight_generating_neural_network = DynamicAttentionNetwork(1, 1, directed_graph)

    optimizer = torch.optim.Adam(weight_generating_neural_network.parameters(), lr=0.01)

    for i in range(1000):
        input_states = torch.rand(5, 1)
        hidden_states = torch.rand(5, 1)
        output_states = weight_generating_neural_network(input_states)
        loss = torch.mean((output_states - input_states)**2)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(output_states)
    print(input_states)
    
    