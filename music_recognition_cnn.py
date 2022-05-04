import torch
from time import time
from torch import nn, optim
from model import MLP_ANN, DatasetGen, CNN
import pandas

"""
Dataframe import and splitting them into traindata and labels
"""
df_train = pandas.read_csv('./regularized_datasets/regularized_train.csv').iloc[:,1:]
df_train_labels = pandas.read_csv('./regularized_datasets/regularized_train_labels.csv').iloc[:,1:]

batch = 112
train_loader = torch.utils.data.DataLoader(DatasetGen([df_train, df_train_labels]),batch_size=batch,shuffle=False)

model_cnn = CNN()

# choose the loss criterion
criterion = nn.CrossEntropyLoss()
music_params, labels = next(iter(train_loader))

# optimizer/learner
optimizer = optim.Adam(model_cnn.parameters(), lr=0.001)
time0 = time() # take note of start time for timing of the process
epochs = 0
mean_loss = 3
old_mean_loss = 0

# train till the cross entropy stabilizes
while abs(old_mean_loss-mean_loss) > 0.00001:
    running_loss = 0
    old_mean_loss = mean_loss
    for music_params, labels in train_loader:

        music_params = torch.reshape(music_params, (batch, 1, 1, 14))
        # Training pass
        optimizer.zero_grad()

        # get output and calculate loss based on chosen criterion
        output = model_cnn(music_params)
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

torch.save(model_cnn.state_dict(), "music_recognition_cnn.sav")