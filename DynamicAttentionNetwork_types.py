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
# NOTE: FOR A PATH OF LENGTH N IN THE NETWORK, YOU HAVE TO RUN THE MODEL FOR AT LEAST
# N STEPS TO PROPAGATE INFORMATION FROM THE FIRST NODE TO THE LAST IN THAT PATH
# TODO:
# 1. Need to make it so that neurons can accept multiple inputs (say multiple channels)
# 3. a version where weights are maintained as states and the relevancies (maybe minus 1)
#   are used to update the weights over time
# 5. implement a hidden state initializer that uses input position
# 7. Try a more complex update function again
# 8. !!!!!I can remove all the nodes from the accumulator update that have no predecessors
# and just make their relevancies all 0 except for their own, which is 1.
# 9. Make it so any number of neuron groups can be created rather than just input,
# inter and output. Should also make it so that there can be multiple kinds of input neurons.

class DynamicAttentionNetwork(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_state_size,
        num_types,
        adjacency_matrix,
        positions=None,
        initialize_hidden_on_data=False,
        device="cpu",
    ):
        super(DynamicAttentionNetwork, self).__init__()
        self.device = device

        self.adjacency_matrix = adjacency_matrix
        self.num_neurons = self.adjacency_matrix.shape[0]
        self.initialize_hidden_on_data = initialize_hidden_on_data
        self.num_types = num_types
        assert (
            sum(self.num_types) == self.num_neurons
        ), "The sum of the number of types must be equal to the number of neurons"

        self.step_size = nn.Parameter(torch.tensor(0.001))

        input_state_size = input_size + hidden_state_size

        step_size_num = hidden_state_size
        if self.num_types[0] != 0:
            self.input_state_update = StateUpdateFunction(
                input_state_size, hidden_state_size
            )
            self.input_state_dependent_time_step = StateDependentTimeStepFunction(hidden_state_size, step_size_num)
        if self.num_types[1] != 0:
            self.inter_state_update = StateUpdateFunction(
                hidden_state_size, hidden_state_size
            )
            self.inter_state_dependent_time_step = StateDependentTimeStepFunction(hidden_state_size, step_size_num)
        if self.num_types[2] != 0:
            self.output_state_update = StateUpdateFunction(
                hidden_state_size, hidden_state_size
            )
            self.output_state_dependent_time_step = StateDependentTimeStepFunction(hidden_state_size, step_size_num)

        self.predecessor_state_accumulator = AccumulatorFunction(
            hidden_state_size, self.adjacency_matrix, self.device
        )
        
        # TODO: I should probably make this an input to the model class, but right now
        # I'm not sure if the parameters will be trainable if I do that
        self.initial_hidden_states=None
        if not self.initialize_hidden_on_data:
            if positions is None:
                self.hidden_state_initializer = BasicHiddenStateInitializer(self.num_neurons, hidden_state_size, device).to(self.device)
                self.init_hidden_states()
            else:
                self.hidden_state_initializer = PositionHiddenStateInitializer(hidden_state_size).to(self.device)
                self.init_hidden_states()
        else:
            self.hidden_state_initializer = InputHiddenStateInitializer(hidden_state_size, self.num_neurons, image_shape=(1, 28, 28), kernel_size=10, device=self.device).to(self.device)

        self.init_params()

    # this function initializes the weights of the model
    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if name == "step_size":
                nn.init.constant_(p, 0.01)
                
    def init_hidden_states(self, input=None):
        if input is None:
            self.initial_hidden_states = self.hidden_state_initializer()
        else:
            self.initial_hidden_states = self.hidden_state_initializer(input)

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
                if self.initialize_hidden_on_data:
                    self.init_hidden_states(input_states)
                    hidden_states = self.initial_hidden_states
                else:
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

        accumulated_predecessor_states = self.predecessor_state_accumulator(
            hidden_states
        )

        updated_hidden_states = self.update_hidden_states(hidden_states, input_states, accumulated_predecessor_states)
        
        return updated_hidden_states
    
    def update_hidden_states(self, hidden_states, input_states, accumulated_predecessor_states):
        updated_input_hidden_states = torch.empty(hidden_states.shape[0], self.num_types[0], hidden_states.shape[2], device=self.device)
        updated_inter_hidden_states = torch.empty(hidden_states.shape[0], self.num_types[1], hidden_states.shape[2], device=self.device)
        updated_output_hidden_states = torch.empty(hidden_states.shape[0], self.num_types[2], hidden_states.shape[2], device=self.device)

        if self.num_types[0] != 0:
            input_accumulated_predecessor_states = accumulated_predecessor_states[
                :, : self.num_types[0], :
            ]
            input_accumulated_predecessor_states = torch.cat((input_accumulated_predecessor_states, input_states), 2)
            input_hidden_states = hidden_states[:, : self.num_types[0], :]
            updated_input_hidden_states = (
                input_hidden_states
                + self.step_size*(self.input_state_dependent_time_step(input_hidden_states))
                * self.input_state_update(input_accumulated_predecessor_states)
            )

        if self.num_types[1] != 0:
            inter_accumulated_predecessor_states = accumulated_predecessor_states[
                :, self.num_types[0] : self.num_types[1], :
            ]
            inter_hidden_states = hidden_states[:, self.num_types[0] : self.num_types[1], :]
            updated_inter_hidden_states = (
                inter_hidden_states
                + self.step_size*self.inter_state_dependent_time_step(inter_hidden_states)
                * self.inter_state_update(inter_accumulated_predecessor_states)
            )

        if self.num_types[2] != 0:
            output_accumulated_predecessor_states = accumulated_predecessor_states[
                :, self.num_types[0] + self.num_types[1] :, :
            ]
            output_hidden_states = hidden_states[:, self.num_types[0] + self.num_types[1] :, :]
            updated_output_hidden_states = (
                output_hidden_states
                + self.step_size*self.output_state_dependent_time_step(output_hidden_states)
                * self.output_state_update(output_accumulated_predecessor_states)
            )
            
        updated_hidden_states = torch.cat(
            (
                updated_input_hidden_states,
                updated_inter_hidden_states,
                updated_output_hidden_states,
            ),
            1,
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
        # can concat QKV and replicate neuron_states to do one big matrix multiply, then separate
        queries = self.Query(neuron_states)
        keys = self.Key(neuron_states)
        values = self.Value(neuron_states)

        relevancies = torch.matmul(queries, keys.transpose(1, 2))
        #relevancies = F.softmax(self.softmax_mask + relevancies, dim=2)
        # normalize the relevancy rows by dividing by the norm of the row
        relevancies = self.adjacency_mask * relevancies
        self.relevancies = relevancies / torch.norm(relevancies, dim=2, keepdim=True)

        accumulated_predecessor_states = torch.matmul(
            self.relevancies, values
        )

        return accumulated_predecessor_states


class StateUpdateFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(StateUpdateFunction, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear1 = nn.Linear(input_size, 100, bias=False)
        self.linear2 = nn.Linear(100, output_size, bias=False)

    def forward(self, input):
        if False:
            x = self.linear1(input)
            x = F.relu(x)
            x = self.linear2(x)
        else:
            x = self.linear(input)
        return x

class StateDependentTimeStepFunction(nn.Module):
    def __init__(self, input_size, output_size):
        super(StateDependentTimeStepFunction, self).__init__()

        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        x = self.linear(input)
        return 2*torch.sigmoid(x) - 1


# TODO: this currently won't work with images that have
# more than 1 channel because of the way the inputs
# are currently shaped in the forward function return.
# This is currently necessary because of how the rest
# of the model uses the input
class InputHiddenStateInitializer(nn.Module):
    # Image shape should be (channels, height, width)
    def __init__(self, hidden_state_size, num_neurons, image_shape, kernel_size=3, device="cpu"):
        super(InputHiddenStateInitializer, self).__init__()

        self.image_shape = image_shape
        self.hidden_state_size = hidden_state_size
        self.num_neurons = num_neurons
        self.device = device

        # a convolution layer that takes in the input image and outputs a hidden state
        # of length hidden_state_size at every pixel
        self.conv = nn.Conv2d(image_shape[0], hidden_state_size, kernel_size, padding="same", bias=False)

    def forward(self, input):
        # reshape the batch of flattened input images to a batch of 2D images based on the image shape
        input = input.view(-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])
        input_hidden_states = self.conv(input).view(-1, self.image_shape[0] * self.image_shape[1] * self.image_shape[2], self.hidden_state_size)
        other_hidden_states = torch.ones(input_hidden_states.shape[0], self.num_neurons - input_hidden_states.shape[1], self.hidden_state_size, device=self.device)
        #mean_input_hidden_states = torch.mean(input_hidden_states, dim=1, keepdim=True)
        #other_hidden_states = mean_input_hidden_states.repeat(1, self.num_neurons - input_hidden_states.shape[1], 1)
        all_hidden_states = torch.cat((input_hidden_states, other_hidden_states), 1)
        return all_hidden_states
    

class BasicHiddenStateInitializer(nn.Module):
    def __init__(self, num_neurons, hidden_state_size, device):
        super(BasicHiddenStateInitializer, self).__init__()
        self.initial_hidden_states = nn.Parameter(torch.zeros(num_neurons, hidden_state_size))
        #self.initial_hidden_states = torch.zeros(num_neurons, hidden_state_size, device=device) #nn.Parameter(torch.zeros(num_neurons, hidden_state_size))

    def forward(self):
        return self.initial_hidden_states


#TODO: position hidden state initializer
# The gradients dont currently flow through the position layer parameters
class PositionHiddenStateInitializer(nn.Module):
    def __init__(self, hidden_state_size, positions, device):
        super(PositionHiddenStateInitializer, self).__init__()
        #self.layer = nn.Linear(3, hidden_state_size, bias=True)
        self.initial_hidden_states = torch.cat((positions, torch.zeros(positions.shape[0], hidden_state_size - positions.shape[1], device=device)), 1)
        
    def forward(self):
        return self.initial_hidden_states
