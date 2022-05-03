import torch
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import pandas

"""
Dataframe import and splitting them into traindata and labels
"""
#train_df = table_to_frame(in_data)
df_train = pandas.read_csv('./regularized_datasets/regularized_train.csv').iloc[:,1:]
df_train_labels = pandas.read_csv('./regularized_datasets/regularized_train_labels.csv').iloc[:,1:]


# df_test = pandas.read_csv('test.csv')
# df_test = df_test.iloc[:,2:-1]
# df_test_labels = pandas.read_csv('test_labels.csv')

class MyDataset(Dataset):

    def __init__(self, data):

        x = data[0].values
        y = data[1].values

        self.x_train = torch.tensor(x, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

train_loader=torch.utils.data.DataLoader(MyDataset([df_train, df_train_labels]),batch_size=256,shuffle=False)
#test_loader=torch.utils.data.DataLoader(MyDataset([df_test, df_test_labels]),batch_size=10,shuffle=True)

# Layer HyperParameters
input_size = 14 # input layer
hidden_sizes = [128,64,32,16] # 4 hidden Layers
output_size = 11 # output layer

"""
Model Declaration
Declare a model with 4 fully connected Layers using the above hyperparams

nn.Linear is a method of defining a fully conected layer from param1 to param2,
the following argument has to be the activation function for that layer.

The general methodology is to follow a Linear Layer Definition with a activation 
function and in the output case, and output function
"""
model_ann = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                  nn.ReLU(),
                  nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                  nn.ReLU(),
                  nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                  nn.ReLU(),
                  nn.Linear(hidden_sizes[2], hidden_sizes[3]),
                  nn.ReLU(),
                  nn.Linear(hidden_sizes[3], output_size),
                  nn.Softmax(dim=1))

# choose the loss criterion
criterion = nn.CrossEntropyLoss()
music_params, labels = next(iter(train_loader))

# optimizer/learner
optimizer = optim.Adam(model_ann.parameters(), lr=0.001)
time0 = time() # take note of start time for timing of the process
epochs = 0
mean_loss = 3

while mean_loss > 0.8:
    running_loss = 0
    for music_params, labels in train_loader:
        # Flatten MNIST images into a 784 long vector
        music_params = music_params.view(music_params.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        # get output and calculate loss based on chosen criterion
        output = model_ann(music_params)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        mean_loss = running_loss/len(train_loader)
        epochs = epochs+1
        print("Epoch {} - Training loss: {}".format(epochs, mean_loss))

timestamp = time() - time0
print("\nTraining Time (in minutes) =", timestamp/60)



