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