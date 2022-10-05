import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import networkx as nx

from DynamicAttentionNetwork_types import DynamicAttentionNetwork
from train_eval_utils import (
    train,
    test,
    self_input_test,
    plot_time_series,
    digraph_to_adjacency,
)

# IMPORTANT NOTE: If n_evals is not at least 2, the model will have to use
# previous input for prediction. This is because the input can't
# get to the output without passing through the hidden state of the input,
# which takes a single time step itself.

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create a network x directed graph with four nodes
G = nx.DiGraph()
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from(
    [(0, 0), (1, 1), (2, 2), (3, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
)

# adjacency_matrix = torch.tensor([[1, 0, 1], [0, 1, 1], [0, 0, 1]], dtype=torch.float32, device=device)
adjacency_matrix = digraph_to_adjacency(G, device=device)

num_neurons = adjacency_matrix.shape[0]

input_size = 1
num_hidden_states = 20
num_types = [2, 0, 2]
n_evals = 2
weight_generating_neural_network = DynamicAttentionNetwork(
    input_size, num_hidden_states, num_types, adjacency_matrix, device=device
).to(device)

for p in weight_generating_neural_network.parameters():
    p.register_hook(lambda grad: grad/(grad.norm() + 1e-6))

optimizer = torch.optim.Adam(weight_generating_neural_network.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 1500], gamma=0.1)

output_indices = torch.tensor([2, 3], device=device).unsqueeze(0).unsqueeze(2)

scale = 20
time_series_length = 100
train_length = 80
time_series = torch.zeros(num_neurons, time_series_length)
x = torch.linspace(0, scale, time_series_length)
time_series[0, :] = 1 * torch.sin(x) + 0.2 * x
time_series[1, :] = 1 * torch.cos(x)

input_order = [0, 1]
target_order = [0, 1]
pure_train_input_time_series = time_series[input_order, :train_length]
pure_train_target_time_series = time_series[target_order, 1 : train_length + 1]

test_input_time_series = time_series[input_order, :-1]
test_target_time_series = time_series[target_order, 1:]

for epoch in range(0, 2000):
    noise = torch.rand(2, 1)#torch.randn_like(train_input_time_series)
    scale = 0
    train_input_time_series = pure_train_input_time_series + scale * noise
    train_target_time_series = pure_train_target_time_series + scale * noise

    train_loss, train_output = train(
        weight_generating_neural_network,
        train_input_time_series,
        train_target_time_series,
        output_indices,
        optimizer,
        n_evals,
        device,
    )
    if epoch % 50 == 0:
        test_length = time_series_length - train_length
        test_loss, test_output = test(
            weight_generating_neural_network,
            test_input_time_series,
            test_target_time_series,
            test_length,
            output_indices,
            n_evals,
            device,
        )
        self_input_test_loss, self_input_test_output = self_input_test(
            weight_generating_neural_network,
            test_input_time_series,
            test_target_time_series,
            test_length,
            output_indices,
            n_evals,
            device,
        )

    print(
        epoch,
        "TRAIN:",
        train_loss,
        "TEST:",
        test_loss,
        "SELF INPUT TEST:",
        self_input_test_loss,
    )

    print(weight_generating_neural_network.predecessor_state_accumulator.relevancies)
    
    scheduler.step()
    

#print the parameters of the model
#for name, param in weight_generating_neural_network.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)

# evaluate the model
test_loss, test_output = test(
    weight_generating_neural_network,
    test_input_time_series,
    test_target_time_series,
    test_length,
    output_indices,
    n_evals,
    device,
)
self_input_test_loss, self_input_test_output = self_input_test(
    weight_generating_neural_network,
    test_input_time_series,
    test_target_time_series,
    test_length,
    output_indices,
    n_evals,
    device,
)

# plot everything
plot_time_series(
    train_input_time_series, train_target_time_series, train_output, "train.png"
)
plot_time_series(
    test_input_time_series[:, -test_length:],
    test_target_time_series[:, -test_length:],
    test_output[:, -test_length:],
    "test.png",
)
plot_time_series(
    test_input_time_series[:, -test_length:],
    test_target_time_series[:, -test_length:],
    self_input_test_output[:, -test_length:],
    "self_input_test.png",
)
