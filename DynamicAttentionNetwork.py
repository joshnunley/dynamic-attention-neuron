import torch
from torch import nn
import numpy as np

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
        abstract_graph,
        hidden_state_size,
        num_evals,
        attention_bounding_method,
        learn_initial_hidden_state=False,
        device="cpu",
    ):
        """
        abstract_graph: a networkx graph object that determines the abstract architecture of the network.
        The nodes of the network are the neuron types and an edge from neuron type one to neuron type two
        means that the neuron type two must consider the states of its own neurons and all
        the neurons in neuron type 2 in the accumulator function.

        Each node contains the following properties (TODO: This needs to be updated):

         - num_neurons: the number of neurons of this type
         - input_size: the number of input states that the neuron type takes in. Can be zero.
         - output_indices: a list of indices of the hidden state that produce the output of the neuron type. Can be empty.
         - state_update_id: an integer identifying the state update function of the neuron type. If state update functions
                            are shared, it is important to ensure both neuron types take the same number of inputs.
         - query_id: an integer identifying the query function of the neuron type.
         - adjacency_mask: a tensor of size (num_neurons, num_predecessor_neurons) that determines
                           the adjacency matrix of the neuron type and the predecessors. Set the adjacency
                           mask to the scalar 1 if the neuron type is connected to all its predecessors.

        Each edge contains the following properties:
         - key_id: an integer identifying the key function that the successor neuron type uses to
                   compute the relevancy of the predecessor neuron type. This includes self-edges.

        hidden_state_size: the size of the hidden state for all neuron types.

        attention_type: a string that determines the type of attention used. Can be "cosine", "softmax", or "norm".
        """
        super(DynamicAttentionNetwork, self).__init__()

        self.abstract_graph = abstract_graph
        self.hidden_state_size = hidden_state_size
        self.num_evals = num_evals
        self.learn_initial_hidden_state = learn_initial_hidden_state
        self.device = device

        # check that attention_bounding_method is selected from the strings "cosine", "softmax", "norm"
        assert attention_bounding_method in ["cosine", "softmax", "norm"]
        self.attention_bounding_method = attention_bounding_method

        self.state_update = nn.ModuleDict()
        self.query = nn.ModuleDict()
        self.key = nn.ModuleDict()
        # TODO: If the state update functions are shared between two neurons,
        # then the state dependent time step should be shared as well.
        self.state_dependent_time_step = nn.ModuleDict()
        self.step_size = nn.Parameter(torch.tensor(1.0))
        self.output_scale = nn.ParameterDict()
        if self.learn_initial_hidden_state:
            self.initial_hidden_state = nn.ParameterDict()

        self.num_neuron_types = 0
        for neuron_type in self.abstract_graph.nodes:
            self.num_neuron_types += 1

            node_properties = self.abstract_graph.nodes[neuron_type]

            assert "num_neurons" in node_properties
            assert "input_size" in node_properties
            assert "output_indices" in node_properties
            assert "state_update_id" in node_properties
            assert "query_id" in node_properties

            num_neurons = node_properties["num_neurons"]
            input_size = node_properties["input_size"]
            output_indices = node_properties["output_indices"]
            state_update_id = node_properties["state_update_id"]
            query_id = node_properties["query_id"]

            state_update = nn.Linear(
                input_size + hidden_state_size, hidden_state_size, bias=False
            )

            if str(state_update_id) in self.state_update.keys():
                if (
                    self.state_update[str(state_update_id)].weight.shape
                    != state_update.weight.shape
                ):
                    raise ValueError(
                        "The input size of neuron type {} is not compatible with state update function with id {}.".format(
                            neuron_type, state_update_id
                        )
                    )
            else:
                self.state_update[str(state_update_id)] = state_update

            if str(query_id) not in self.query.keys():
                self.query[str(query_id)] = nn.Linear(
                    hidden_state_size, hidden_state_size, bias=False
                )
            str_neuron_type = str(neuron_type)
            if output_indices:
                self.output_scale[str_neuron_type] = nn.Parameter(torch.tensor(1.0))

            self.state_dependent_time_step[str_neuron_type] = nn.Linear(
                hidden_state_size, hidden_state_size
            )
            if learn_initial_hidden_state:
                self.initial_hidden_state[str_neuron_type] = nn.Parameter(
                    torch.zeros(num_neurons, hidden_state_size)
            )

        for edge in abstract_graph.edges:
            edge_properties = abstract_graph.edges[edge]

            assert "key_id" in edge_properties
            if self.abstract_graph.nodes[edge[1]]["adjacency_masks_provided"]:
                assert "adjacency_mask" in edge_properties

            key_id = abstract_graph.edges[edge]["key_id"]
            if str(key_id) not in self.key.keys():
                self.key[str(key_id)] = nn.Linear(
                    input_size + hidden_state_size, hidden_state_size, bias=False
                )

        self.init_params()

    # this function initializes the weights of the model
    def init_params(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            if "step_size" in name:
                nn.init.constant_(p, 1)
            if "initial_hidden_state" in name:
                nn.init.normal_(p, 0, 0.1)
            if "output_scale" in name:
                nn.init.constant_(p, 10)
            #if "state_update" in name:
            #    1/np.power(self.num_evals, 2)*nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_states,
        hidden_states=None,
        initial_output_mask=None,
        initial_outputs=None,
    ):
        # NOTE: This doesn't currently cover the case where
        # the initial hidden states are provided and
        # initial outputs are provided, but I'm not sure
        # it makes sense to do that.
        if hidden_states is None:
            if initial_outputs is None:
                hidden_states = self.initial_hidden_state
            else:
                hidden_states = self.initial_hidden_state
                hidden_states = hidden_states * initial_output_mask + initial_outputs

        # make batches work
        # this could probably be made more efficient
        for neuron_type in self.abstract_graph.nodes:
            if neuron_type in input_states.keys():
                if input_states[neuron_type].dim() == 2:
                    batch_size = 1
                    input_states[neuron_type] = torch.unsqueeze(
                        input_states[neuron_type], 0
                    )
                elif input_states[neuron_type].dim() == 3:
                    batch_size = input_states[neuron_type].shape[0]

        for neuron_type in self.abstract_graph.nodes:
            if self.learn_initial_hidden_state:
                hidden_state_neuron_type = str(neuron_type)
            else:
                hidden_state_neuron_type = neuron_type

            # TODO: Try to produce a minimal example that shows that expand
            # doesn't reference the data properly when nn.Parameters is
            # contained in a ParameterDict, while it does when it is
            # not contained in a ParameterDict.
            if hidden_states[hidden_state_neuron_type].dim() == 2:
                hidden_states[hidden_state_neuron_type] = torch.unsqueeze(
                    hidden_states[hidden_state_neuron_type], 0
                )
                hidden_states[hidden_state_neuron_type] = hidden_states[hidden_state_neuron_type].expand(
                    batch_size, -1, -1
                )
            elif hidden_states[hidden_state_neuron_type].shape[0] != batch_size:
                hidden_states[hidden_state_neuron_type] = hidden_states[
                    hidden_state_neuron_type
                ][0, :, :].expand(batch_size, -1, -1)

        accumulated_predecessor_states = self.accumulate_predecessor_states(
            hidden_states
        )

        updated_hidden_states = self.update_hidden_states(
            hidden_states, input_states, accumulated_predecessor_states
        )

        outputs = {}
        for neuron_type in self.abstract_graph.nodes:
            if self.learn_initial_hidden_state:
                hidden_state_neuron_type = str(neuron_type)
            else:
                hidden_state_neuron_type = neuron_type

            ouput_indices = self.abstract_graph.nodes[neuron_type]["output_indices"]

            if ouput_indices:
                outputs[neuron_type] = self.output_scale[str(neuron_type)] * updated_hidden_states[hidden_state_neuron_type][
                    :, :, self.abstract_graph.nodes[neuron_type]["output_indices"]
                ]
        
        return outputs, updated_hidden_states

    def update_hidden_states(
        self, hidden_states, input_states, accumulated_predecessor_states
    ):
        updated_hidden_states = {}
        for neuron_type in self.abstract_graph.nodes:
            if self.learn_initial_hidden_state:
                hidden_state_neuron_type = str(neuron_type)
            else:
                hidden_state_neuron_type = neuron_type

            str_neuron_type = str(neuron_type)
            node_properties = self.abstract_graph.nodes[neuron_type]

            state_update_id = str(node_properties["state_update_id"])
            input_size = node_properties["input_size"]

            if input_size > 0:
                incoming_neuron_states = torch.cat(
                    (
                        input_states[neuron_type],
                        accumulated_predecessor_states[neuron_type],
                    ),
                    dim=2,
                )
            else:
                incoming_neuron_states = accumulated_predecessor_states[neuron_type]

            updated_hidden_states[hidden_state_neuron_type] = hidden_states[hidden_state_neuron_type] + self.step_size * (
                2
                * torch.sigmoid(
                    self.state_dependent_time_step[str_neuron_type](
                        hidden_states[hidden_state_neuron_type]
                    )
                )
                - 1
            ) * self.state_update[
                state_update_id
            ](
                incoming_neuron_states
            )

        return updated_hidden_states

    # This function computes the accumulated predecessor states for each neuron type
    # using the abstract graph to determine the predecessors of each neuron type and the
    # queries and keys to compute self relevancies and relevancy of predecessor neuron types.
    def accumulate_predecessor_states(self, hidden_states):
        accumulated_predecessor_states = {}

        def get_cosine_relevancies(queries, keys, adjacency_mask):
            queries_norm = torch.norm(queries, dim=2, keepdim=True)
            keys_norm = torch.norm(keys, dim=2, keepdim=True)
            
            if adjacency_mask.dim() != 0:
                scale = 1/torch.sum(adjacency_mask, dim=2, keepdim=True)
            else:
                scale = 1/keys.shape[1]

            relevancies = (
                scale
                * adjacency_mask
                * torch.matmul(
                    queries / queries_norm, (keys / keys_norm).transpose(1, 2)
                )
            )

            return relevancies 

        def get_softmax_relevancies(queries, keys, adjacency_mask):
            softmax_mask = -1e9 * (1 - adjacency_mask)

            relevancies = torch.matmul(queries, keys.transpose(1, 2))
            relevancies = adjacency_mask * (
                torch.nn.functional.softmax(relevancies + softmax_mask, dim=2)
            )
            return relevancies

        def get_norm_relevancies(queries, keys, adjacency_mask):
            relevancies = adjacency_mask * torch.matmul(queries, keys.transpose(1, 2))
            relevancies = relevancies / torch.norm(relevancies, dim=2, keepdim=True)
            return relevancies

        for neuron_type in self.abstract_graph.nodes:
            if self.learn_initial_hidden_state:
                hidden_state_neuron_type = str(neuron_type)
            else:
                hidden_state_neuron_type = neuron_type            
            
            if not list(self.abstract_graph.pred[neuron_type]):
                accumulated_predecessor_states[neuron_type] = hidden_states[
                    hidden_state_neuron_type
                ]
                self.relevancies = 1
                continue

            node_properties = self.abstract_graph.nodes[neuron_type]

            query_id = str(node_properties["query_id"])
            adjacency_mask_provided = node_properties["adjacency_masks_provided"]

            queries = self.query[query_id](hidden_states[hidden_state_neuron_type])

            # Loop through the neurons predecessors and concatenate the keys
            keys = torch.empty(
                hidden_states[hidden_state_neuron_type].shape[0],
                0,
                hidden_states[hidden_state_neuron_type].shape[2],
                device=self.device,
            )
            predecessor_hidden_states = torch.empty(
                hidden_states[hidden_state_neuron_type].shape[0],
                0,
                hidden_states[hidden_state_neuron_type].shape[2],
                device=self.device,
            )
            adjacency_mask = torch.empty(
                1,
                hidden_states[hidden_state_neuron_type].shape[1],
                0,
                device=self.device,
            )
            for predecessor_type in self.abstract_graph.predecessors(neuron_type):
                if self.learn_initial_hidden_state:
                    predecessor_hidden_state_neuron_type = str(predecessor_type)
                else:
                    predecessor_hidden_state_neuron_type = predecessor_type

                edge_properties = self.abstract_graph.edges[predecessor_type, neuron_type]

                if adjacency_mask_provided:
                    edge_adjacency_mask = torch.unsqueeze(edge_properties["adjacency_mask"], 0)
                    adjacency_mask = torch.cat((adjacency_mask, edge_adjacency_mask), dim=2)

                key_id = str(
                    edge_properties["key_id"]
                )
                keys = torch.cat(
                    (keys, self.key[key_id](hidden_states[predecessor_hidden_state_neuron_type])), dim=1
                )

                predecessor_hidden_states = torch.cat(
                    (predecessor_hidden_states, hidden_states[predecessor_hidden_state_neuron_type]), dim=1
                )

                
            if adjacency_mask_provided:
                adjacency_mask = adjacency_mask.expand((hidden_states[hidden_state_neuron_type].shape[0], -1, -1))
            else:
                adjacency_mask = torch.tensor(1)

            if self.attention_bounding_method == "cosine":
                self.relevancies = get_cosine_relevancies(
                    queries, keys, adjacency_mask
                )
            elif self.attention_bounding_method == "softmax":
                self.relevancies = get_softmax_relevancies(
                    queries, keys, adjacency_mask
                )
            elif self.attention_bounding_method == "norm":
                self.relevancies = get_norm_relevancies(queries, keys, adjacency_mask)

            accumulated_predecessor_states[neuron_type] = torch.matmul(
                self.relevancies, predecessor_hidden_states
            )

        return accumulated_predecessor_states
