import torch
import torch.nn.functional as F
import networkx as nx

import matplotlib.pyplot as plt


def train(
    model, input_time_series, target_time_series, output_indices, optimizer, n_evals, device
):
    model.train()
    
    output = torch.zeros(target_time_series.shape, device=device)
    time_series_length = input_time_series.shape[1]
    initial_input = input_time_series[:, 0].reshape(1, -1).to(device)
    hidden_state = model(initial_input)
    output[:, 0] = hidden_state.gather(1, output_indices).flatten()
    for t in range(1, time_series_length):
        input = input_time_series[:, t].reshape(1, -1).to(device)
        for i in range(n_evals):
            hidden_state = model(input, hidden_state)
        output[:, t] = hidden_state.gather(1, output_indices).flatten()

    loss = F.mse_loss(output, target_time_series)
    loss.backward()
    # print the gradients of the model
    #for name, param in model.named_parameters():
    #    if param.requires_grad:
    #        print(name, param.grad)

    optimizer.step()
    optimizer.zero_grad()
    return loss, output


def test(
    model, input_time_series, target_time_series, test_length, output_indices, n_evals, device
):
    model.eval()

    with torch.no_grad():
        output = torch.zeros(target_time_series.shape, device=device)
        time_series_length = input_time_series.shape[1]
        initial_input = input_time_series[:, 0].reshape(1, -1).to(device)
        hidden_state = model(initial_input)
        output[:, 0] = hidden_state.gather(1, output_indices).flatten()
        for t in range(1, time_series_length):
            input = input_time_series[:, t].reshape(1, -1).to(device)
            for i in range(n_evals):
                hidden_state = model(input, hidden_state)
            output[:, t] = hidden_state.gather(1, output_indices).flatten()

    loss = F.mse_loss(
        output[:, -test_length :],
        target_time_series[:, -test_length :],
    )
    return loss, output


def self_input_test(
    model,
    input_time_series,
    target_time_series,
    self_input_length,
    output_indices,
    n_evals,
    device,
):
    model.eval()

    with torch.no_grad():
        output = torch.zeros(target_time_series.shape, device=device)
        train_time_series_length = input_time_series.shape[1] - self_input_length
        initial_input = input_time_series[:, 0].reshape(1, -1).to(device)
        hidden_state = model(initial_input)
        output[:, 0] = hidden_state.gather(1, output_indices).flatten()
        for t in range(1, train_time_series_length):
            input = input_time_series[:, t].reshape(1, -1).to(device)
            for i in range(n_evals):
                hidden_state = model(input, hidden_state)
            output[:, t] = hidden_state.gather(1, output_indices).flatten()

        for t in range(
            train_time_series_length, train_time_series_length + self_input_length
        ):
            input = input_time_series[:, t].reshape(1, -1).clone().to(device)
            input[0, :] = output[:, t - 1]
            #input[0, :] = torch.rand(input.shape[1], device=device)
            for i in range(n_evals):
                hidden_state = model(input, hidden_state)
            output[:, t] = hidden_state.gather(1, output_indices).flatten()

    loss = F.mse_loss(
        output[:, -self_input_length:],
        target_time_series[:, -self_input_length:],
    )
    return loss, output


def digraph_to_adjacency(directed_graph, device):
    # create a numpy array of zeros with the same shape as the adjacency matrix
    # and then fill in the ones
    adjacency_matrix = nx.to_numpy_array(directed_graph)
    adjacency_matrix = torch.tensor(
        adjacency_matrix, dtype=torch.float32, device=device
    )
    return adjacency_matrix


def plot_time_series(input_time_series, target_time_series, output, file_name):
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(input_time_series[0, :].cpu().detach().numpy(), color="red")
    ax[0].plot(target_time_series[0, :].cpu().detach().numpy(), color="blue")
    ax[0].plot(output[0, :].cpu().detach().numpy(), color="orange")
    ax[1].plot(input_time_series[1, :].cpu().detach().numpy(), color="red")
    ax[1].plot(target_time_series[1, :].cpu().detach().numpy(), color="blue")
    ax[1].plot(output[1, :].cpu().detach().numpy(), color="orange")

    plt.savefig(file_name)
