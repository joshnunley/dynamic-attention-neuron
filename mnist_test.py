# test the Dynamic Attention Network on the MNIST dataset

from tkinter import W
import torch
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from DynamicAttentionNetwork_types import DynamicAttentionNetwork

# TODO: 
# 1. How does graph structure affect the results? Does it help to have the grid
# connecting the input together? Or can we just get away with edges from the input
# to the output?
# 2. Would it help to have inter neurons that receive the input, connect amongst
# themselves, and then connect to the output? This would allow the input
# and interneurons to have different update functions, but would double the number
# of neurons.
# 3. take a trained model and evolve the adjacency matrix to maximize accuracy
# 4. visualize the relevancy row of the output neuron as an image
# 5. If I provide an adjacency matrix of all ones, is there a dynamic graph structure
# that naturally arises in the relevancies?
# 6. This is crazy, but would it be possible to dynamically change the number of
# neurons necessary for each prediction? It would be possible to, for example,
# only keep the neurons that have one as input and compute a nearest neighbor 
# graph to maintain some sort of topology. Or this would allow for subsampling,
# or for A FOVEA

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
# load the MNIST dataset
mnist_train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# create a directed graph with 784 nodes
# This graph is a grid so each node has 4 predecessors
# and 4 successors, and one extra node that has every other
# node as a predecessor, and no successors.
grid_size = (28, 28)
directed_graph = nx.grid_2d_graph(grid_size[0], grid_size[1]).to_directed()
directed_graph.add_node(grid_size)
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        directed_graph.add_edge((i, j), grid_size)

# add self loops to all nodes
for node in directed_graph.nodes:
    directed_graph.add_edge(node, node)

print(directed_graph.number_of_nodes())
# create a function that takes in a network x directed graph
# and a device and returns an adjacency matrix converted to a 
# tensorbool on the device
def digraph_to_tensorbool(directed_graph, device):
    # create a numpy array of zeros with the same shape as the adjacency matrix
    # and then fill in the ones
    adjacency_matrix = nx.to_numpy_array(directed_graph)
    adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32, device=device)
    return adjacency_matrix
        
# convert the directed graph to a tensorbool
adjacency_matrix = digraph_to_tensorbool(directed_graph, device=device)
#adjacency_matrix = torch.ones(adjacency_matrix.shape, device=device)



# compute a 2d array of the pixel coordinates
# centered around the origin
# TODO: Turn this into a more general function
x_size = grid_size[0]
y_size = grid_size[1]
# This is the number of layers, one input layer and one output layer
z_size = 2
x = torch.linspace(-1, 1, x_size)
y = torch.linspace(-1, 1, y_size)
x, y = torch.meshgrid(x, y)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = -1*torch.ones(x.shape).reshape(-1, 1)
positions = torch.cat((x, y, z), dim=1)
# add the output position
positions = torch.cat((positions, torch.tensor([0, 0, 1]).reshape(1, 3)), dim=0)
positions = positions.to(device)
print(positions.shape)




input_size = 1
hidden_state_size = 40
num_types = [784, 0, 1]
n_evals = 9
dan = DynamicAttentionNetwork(
    input_size, hidden_state_size, num_types, adjacency_matrix, positions=None, initialize_hidden_on_data=False, device=device
).to(device)


clip_value = 1
for p in dan.parameters():
    p.register_hook(lambda grad: grad/(grad.norm() + 1e-6))

# create a data loader for the MNIST dataset
# Right now because of the structure of the model I can't increase the batch_size
batch_size = 32
test_batch_size = 32
mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=test_batch_size, shuffle=True, drop_last=True)


optimizer = optim.Adam(dan.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 3, 4], gamma=0.1)

#print the parameters of the model nicely
for name, param in dan.named_parameters():
    print(name, param.shape)
    



# train the Dynamic Attention Network
test_accuracy = -1
for epoch in range(8):
    partial_train_accuracy = 0
    train_correct = 0
    for i, (images, labels) in enumerate(mnist_train_loader):
        labels = labels.to(device)
        
        optimizer.zero_grad()
        input_states = images.view(batch_size, 28*28).to(device)

        # will this still learn with random initial states?
        #hidden_states = torch.ones(adjacency_matrix.shape[0], hidden_state_size).to(device)
        hidden_states = dan(input_states)
        #hidden_states = 0.01*torch.zeros(adjacency_matrix.shape[0], hidden_state_size).to(device)
        for j in range(n_evals):
            hidden_states = dan(input_states, hidden_states)

        output = hidden_states[:, -1, -10:]
        
        # the outputs represent the probability of the image being a digit
        # so we use log softmax to get the log probability of the correct label
        loss = F.nll_loss(F.log_softmax(output, dim=1), labels)
        
        train_correct += (output.argmax(dim=1) == labels).sum().item()
        partial_train_accuracy = train_correct / ((i+1)*batch_size)

        loss.backward()
        #torch.nn.utils.clip_grad_norm_(dan.parameters(), 5)
        optimizer.step()

        #for name, param in dan.named_parameters():
        #    print(name, param.grad)
        if (i+1) % 100 == 0:
            print("Epoch: {}, Batch: {}, Loss: {}, Partial Train Accuracy: {}, Test Accuracy: {}".format(epoch, i, loss.item(), partial_train_accuracy, test_accuracy))
        if (i+1) % 1000 == 0:
            with torch.no_grad():
                # calculate the accuracy of the model on a single batch
                # of the test data
                # store a time series of the relevancies
                relevancy_test_size = 10
                relevancy_time_series = torch.zeros(relevancy_test_size, adjacency_matrix.shape[0], adjacency_matrix.shape[1], n_evals+1)

                correct = 0
                total = 0
                images, labels = next(iter(mnist_test_loader))
                labels = labels[:relevancy_test_size].to(device)
                input_states = images[:relevancy_test_size].view(relevancy_test_size, 28*28).to(device)

                hidden_states = dan(input_states)
                for j in range(n_evals):
                    hidden_states = dan(input_states, hidden_states)
                    relevancy_time_series[:, :, :, j] = dan.predecessor_state_accumulator.relevancies

                output = hidden_states[:, -1, -10:]
                predicted = torch.argmax(output, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                partial_test_accuracy = correct/total
        
                # checkpoint the model
                torch.save(dan.state_dict(), "./saved_data/checkpoint_dan" + str(hidden_state_size) + "_" + str(n_evals) + ".pt")
                # save the relevancy time series
                torch.save(relevancy_time_series, "./saved_data/relevancy_time_series" + str(hidden_state_size) + "_" + str(n_evals) + ".pt")
                torch.save(images, "./saved_data/test_images" + str(hidden_state_size) + "_" + str(n_evals) + ".pt")
            
    with torch.no_grad():
        print("Calculating accuracy on entire test set...") 
        # Print the accuracy of the model on the entire test dataset
        correct = 0
        total = 0
        for images, labels in mnist_test_loader:
            labels = labels.to(device)
            input_states = images.view(test_batch_size, 28*28).to(device)
            hidden_states = dan(input_states)

            #hidden_states = 0.01*torch.randn(adjacency_matrix.shape[0], hidden_state_size).to(device)
            for j in range(n_evals):
                hidden_states = dan(input_states, hidden_states)

            output = hidden_states[:, -1, -10:]
            predicted = torch.argmax(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        test_accuracy = correct/total
        print(test_accuracy)

    scheduler.step()
    
# print all the model parameters
for name, param in dan.named_parameters():
    print(name, param)