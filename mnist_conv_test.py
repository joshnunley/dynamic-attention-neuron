# this file tests a simple convolutional neural network on the mnist dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.mean(x, dim=(2, 3))
        return F.log_softmax(x, dim=1)
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the mnist dataset
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # create a data loader for the mnist dataset
    batch_size = 32
    mnist_train_loader = DataLoader(mnist_train_dataset, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test_dataset, batch_size=100, shuffle=True)

    # create a convolutional neural network
    cnn = Net()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    # train the convolutional neural network
    for epoch in range(8):
        for i, (images, labels) in enumerate(mnist_train_loader):
            labels = labels.to(device)

            optimizer.zero_grad()
            output = cnn(images)

            # the outputs represent the probability of the image being a digit
            # so we use log softmax to get the log probability of the correct label
            loss = F.nll_loss(output, labels)

            loss.backward()
            optimizer.step()

            print("Epoch: {}, Batch: {}, Loss: {}".format(epoch, i, loss.item()))

        # calculate the accuracy of the model on a single batch
        # of the test data
        test_batch_size = 100
        test_batch = next(iter(mnist_test_loader))
        test_images = test_batch[0].to(device)
        test_labels = test_batch[1].to(device)

        with torch.no_grad():
            test_output = cnn(test_images)
            test_loss = F.nll_loss(test_output, test_labels)
            test_pred = test_output.argmax(dim=1, keepdim=True)
            test_accuracy = test_pred.eq(test_labels.view_as(test_pred)).sum().item() / test_batch_size

        print("Epoch: {}, Test Loss: {}, Test Accuracy: {}".format(epoch, test_loss.item(), test_accuracy))