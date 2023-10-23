import torch
import networkx as nx
import numpy as np
from copy import deepcopy

from DynamicAttentionNetwork import DynamicAttentionNetwork
from graph_structures import get_abstract_graph


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

hidden_state_size = 30
n_evals = 15
graph_type = 'basic_fully_connected_input'
properties_for_nodes, properties_for_edges, output_id = get_abstract_graph(graph_type, device)
print('graph_type: ', graph_type, "number of evals: ", n_evals)

grid_size = (28, 28)
# compute a 2d array of the pixel coordinates
# centered around the origin
# TODO: Turn this into a more general function
x_size = grid_size[0]
y_size = grid_size[1]
z_size = len(properties_for_nodes)
# This is the number of layers, one input layer and one output layer
x = torch.linspace(-1, 1, x_size)
y = torch.linspace(-1, 1, y_size)
z = torch.linspace(-1, 1, z_size)
x, y, z = torch.meshgrid(x, y, z)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(-1, 1)
print(x.shape)
#z = -1*torch.ones(x.shape).reshape(-1, 1)
positions = torch.cat((x, y, z), dim=1).reshape(grid_size[0]*grid_size[1], len(properties_for_nodes), 3)
print(positions.shape)
# add the output position
#positions = torch.cat((positions, torch.tensor([0, 0, 1]).reshape(1, 3)), dim=0)
positions = positions.to(device)


# multiply initial_hidden_state by 1/sqrt(hidden_state_size) as outlined in
# https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf (1 / np.sqrt(hidden_state_size))
abstract_graph = nx.DiGraph()
initial_hidden_states = {}
for node_id, properties in enumerate(properties_for_nodes):
    abstract_graph.add_node(node_id, **properties)

    #if not properties["output_indices"]:
    #    initial_hidden_states[node_id] = 0.1 * torch.randn(
    #        properties["num_neurons"], hidden_state_size
    #    ).to(device)
    #else:
    #    initial_hidden_states[node_id] = 0.1*torch.ones(
    #        properties["num_neurons"], hidden_state_size
    #    ).to(device)
    if node_id != output_id:
        node_positions = positions[:, node_id, :]
        initial_hidden_states[node_id] = torch.cat((node_positions, torch.zeros(node_positions.shape[0], hidden_state_size - node_positions.shape[1], device=device)), 1)
    else:
        initial_hidden_states[node_id] = 0.1 * torch.randn(
            properties["num_neurons"], hidden_state_size
        ).to(device)
        initial_hidden_states[node_id][:, 0] = 0
        #torch.cat((positions[784:, :], torch.zeros(positions[784:, :].shape[0], hidden_state_size - positions[784:, :].shape[1], device=device)), 1)
    
for edge, properties in properties_for_edges.items():
    abstract_graph.add_edge(*edge, **properties)

learn_initial_hidden_state = False
model = DynamicAttentionNetwork(
    abstract_graph=abstract_graph,
    hidden_state_size=hidden_state_size,
    num_evals=n_evals,
    attention_bounding_method="cosine",
    learn_initial_hidden_state=learn_initial_hidden_state,
    device=device,
).to(device)

print(
    "Parameter Count: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)


# print the prameters and their names
for name, param in model.named_parameters():
    print(name, param.shape)


# train the model on the mnist dataset

# get the data
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
# load the MNIST dataset
mnist_train_dataset = datasets.MNIST(
    root="./data", train=True, transform=transform, download=True
)
mnist_test_dataset = datasets.MNIST(
    root="./data", train=False, transform=transform, download=True
)


num_param_sets = 0
for p in model.parameters():
    num_param_sets += 1

clip_value = 1
for p in model.parameters():
    p.register_hook(
        lambda grad: grad / (grad.norm() + 1e-8)
    )  # if grad.norm() > clip_value else grad)

# create a data loader for the MNIST dataset
batch_size = 30
test_batch_size = 30
num_gradient_accumulations = 2
mnist_train_loader = DataLoader(
    mnist_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
)
mnist_test_loader = DataLoader(
    mnist_test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True
)

optimizer = optim.Adam(model.parameters(), lr=0.01)
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8, 12], gamma=0.1)
#I think gamma should be 0.8 for lower number of epochs and 0.9 for higher number of epochs
# it's possible that 0.9 is just better
gamma = 0.9
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)
print("batch size: ", batch_size, "num_gradient_accumulations: ", num_gradient_accumulations, "optimizer: ", optimizer, "scheduler gamma: ", gamma)

# train the Dynamic Attention Network
test_accuracy = -1
optimizer.zero_grad()
for epoch in range(30):
    partial_train_accuracy = 0
    train_correct = 0
    for i, (images, labels) in enumerate(mnist_train_loader):
        labels = labels.to(device)
        input_states = {}
        input_states[0] = images.view(batch_size, 28 * 28, 1).to(device)

        if learn_initial_hidden_state:
            outputs, hidden_states = model(
                input_states
            )
        else:
            outputs, hidden_states = model(
                input_states, deepcopy(initial_hidden_states)
                )
        

        for j in range(n_evals):
            outputs, hidden_states = model(input_states, hidden_states)

        output = outputs[output_id][:, :, 0]

        # TODO: Add an output scale parameter to every node that
        # has an output.
        loss = F.nll_loss(F.log_softmax(100*output, dim=1), labels) / num_gradient_accumulations

        train_correct += (output.argmax(dim=1) == labels).sum().item()
        partial_train_accuracy = train_correct / ((i + 1) * batch_size)

        loss.backward()
        if (((i + 1) % num_gradient_accumulations) == 0) or (i + 1 == len(mnist_train_loader)):
            optimizer.step()
            optimizer.zero_grad()

        # for name, param in dan.named_parameters():
        #    print(name, param.grad)
        if (i + 1) % (100*num_gradient_accumulations) == 0:
            print(
                "Epoch: {}, Batch: {}, Loss: {}, Partial Train Accuracy: {}, Test Accuracy: {}".format(
                    epoch, int(i/num_gradient_accumulations), loss.item()*num_gradient_accumulations, partial_train_accuracy, test_accuracy
                )
            )
        if (i + 1) % 1000 == 0:
            with torch.no_grad():
                # calculate the accuracy of the model on a single batch
                # of the test data
                # store a time series of the relevancies
                relevancy_test_size = 10
                relevancy_time_series = []
                hidden_state_time_series = []

                correct = 0
                total = 0
                images, labels = next(iter(mnist_test_loader))
                labels = labels[:relevancy_test_size].to(device)
                input_states[0] = (
                    images[:relevancy_test_size]
                    .view(relevancy_test_size, 28 * 28, 1)
                    .to(device)
                )

                if learn_initial_hidden_state:
                    outputs, hidden_states = model(
                        input_states
                    )
                else:
                    outputs, hidden_states = model(
                        input_states, deepcopy(initial_hidden_states)
                        )
                hidden_state_time_series.append(hidden_states)
                relevancy_time_series.append(model.relevancies)
                for j in range(n_evals):
                    outputs, hidden_states = model(input_states, hidden_states)
                    hidden_state_time_series.append(hidden_states)
                    relevancy_time_series.append(model.relevancies)

                output = outputs[output_id][:, :, 0]
                predicted = torch.argmax(output, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                partial_test_accuracy = correct / total

                # checkpoint the model
                torch.save(
                    model.state_dict(),
                    "./saved_data/checkpoint_dan"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )
                # save the relevancy time series
                torch.save(
                    relevancy_time_series,
                    "./saved_data/relevancy_time_series"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )
                torch.save(
                    hidden_state_time_series,
                    "./saved_data/hidden_state_time_series"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )
                torch.save(
                    images,
                    "./saved_data/test_images"
                    + str(hidden_state_size)
                    + "_"
                    + str(n_evals)
                    + "_"
                    + graph_type
                    + "_classify.pt",
                )

    with torch.no_grad():
        print("Calculating accuracy on entire test set...")
        # Print the accuracy of the model on the entire test dataset
        correct = 0
        total = 0
        for images, labels in mnist_test_loader:
            labels = labels.to(device)
            input_states[0] = images.view(test_batch_size, 28 * 28, 1).to(device)

            if learn_initial_hidden_state:
                outputs, hidden_states = model(
                    input_states
                )
            else:
                outputs, hidden_states = model(
                    input_states, deepcopy(initial_hidden_states)
                    )
            for j in range(n_evals):
                outputs, hidden_states = model(input_states, hidden_states)

            output = outputs[output_id][:, :, 0]
            predicted = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        print(test_accuracy)

    scheduler.step()

# print all the model parameters
for name, param in model.named_parameters():
    print(name, param)

# Print the parameter count
print(
    "Parameter Count: {}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
)
print("Test Accuracy: {}".format(test_accuracy))
