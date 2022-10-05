import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# TODO: Look at the change in the last ten parameters of the output neuron over time
# to see how the neuron changes it's classification over time

# load the model, relevancy time series and the image data
hidden_state_size = 40
n_evals = 9
dan = torch.load("./saved_data/checkpoint_dan" + str(hidden_state_size) + "_" + str(n_evals) + ".pt", map_location=torch.device("cpu"))
relevancy_time_series = torch.load("./saved_data/relevancy_time_series" + str(hidden_state_size) + "_" + str(n_evals) + ".pt", map_location=torch.device("cpu"))
image_data = torch.load("./saved_data/test_images" + str(hidden_state_size) + "_" + str(n_evals) + ".pt", map_location=torch.device("cpu"))

# get the relevancy row for the output neuron
output_neuron_relevancies = relevancy_time_series[:, -1, :, :]

number = 0
num_time_steps = n_evals
image_output_neuron_relevancies = output_neuron_relevancies[number, :, :]

fig, ax = plt.subplots(2, 1)
def animate(i):
    ax[0].set_title("Relevancy: " + str(i))
    ax[1].set_title("Image")

    ax[0].imshow(image_output_neuron_relevancies[:-1, i].reshape(28, 28))
    ax[1].imshow(image_data[number, :].reshape(28, 28))
    return ax

anim = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=1000)
anim.save("./plots/output_relevancies.gif", writer="Pillow")


fig, ax = plt.subplots()
def animate(i):
    ax.set_title("Relevancy: " + str(i))
    ax.imshow(relevancy_time_series[number, :, :, i])
    return ax

anim = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=1000)
anim.save("./plots/relevancies.gif", writer="Pillow")


fig, ax = plt.subplots()
def animate(i):
    ax.set_title("Relevancy: " + str(i))
    ax.imshow(torch.mean(relevancy_time_series[number, :, :-1, i], dim=0).reshape(28, 28))
    return ax

anim = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=1000)
anim.save("./plots/mean_predecessor_relevancies.gif", writer="Pillow")

fig, ax = plt.subplots()
def animate(i):
    ax.set_title("Relevancy: " + str(i))
    ax.imshow(relevancy_time_series[number, :-1, -1, i].reshape(28, 28))
    return ax

anim = animation.FuncAnimation(fig, animate, frames=num_time_steps, interval=1000)
anim.save("./plots/output_relevancies_to_others.gif")


plt.figure()
plt.plot(dan.initial_hidden_states.detach())
plt.savefig("./plots/initial_hidden_states.png")