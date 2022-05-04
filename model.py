import torch
from torch import nn
from torch.utils.data import Dataset

# Model HyperParams
hidden_layers = 4
hidden_layer_size = 64
input_size = 14
output_size = 11

class MLP_ANN(nn.Module):
    def __init__(self):
        super(MLP_ANN, self).__init__()

        # define the input hidden layer
        self.input = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.ReLU() # activation function
        )

        # definition for a fully connected hidden layer
        self.hidden = nn.Sequential(
            nn.Linear(hidden_layer_size,hidden_layer_size),
            nn.ReLU()
        )

        # the final output layer
        self.out = nn.Sequential(
            nn.Linear(hidden_layer_size, output_size),
            nn.Softmax(dim=1)# activation function
        )

    def forward(self, x):
        x = self.input(x)
        for i in range(hidden_layers):
            x = self.hidden(x)

        output = self.out(x)
        return output   # return x for visualization

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # 1 input channel
                out_channels=16, # 16 output channels
                kernel_size=(5,5), # sliding window matrix of size (5,5)
                stride=(1,1), # Move the window by the stride value
            ),
            nn.ReLU(), # activation function
            nn.MaxPool2d(kernel_size=2), # pooling function with window matrix of size (2,2)
        )

        # same as above but few changes to input and output layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (5,5), (1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.out = nn.Linear(32 * 7 * 7, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x  # return x for visualization

class DatasetGen(Dataset):
    def __init__(self, data):

        x = data[0].values
        y = data[1].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]