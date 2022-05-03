import torch
from time import time
from torch import nn, optim
from torch.utils.data import Dataset
from Orange.data.pandas_compat import table_from_frame, table_to_frame
import pandas

df_test = pandas.read_csv('./regularized_datasets/regularized_test2.csv').iloc[:,1:]
df_test_labels = pandas.read_csv('./regularized_datasets/regularized_test_labels2.csv').iloc[:,1:]

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


test_loader=torch.utils.data.DataLoader(MyDataset([df_test, df_test_labels]),batch_size=256,shuffle=False)

input_size = 14 # input layer
hidden_sizes = [128,64,32,16] # 4 hidden Layers
output_size = 11 # output layer

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

model_ann.load_state_dict(torch.load("music_recognition_mlp.sav"))
model_ann.eval()

correct_count = 0
all_count = 0

for music_params, labels in test_loader:
    for i in range(len(labels)):
        music_params = music_params.view(music_params.shape[0], -1)
        with torch.no_grad():
            logps = model_ann(music_params)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = list(labels.numpy()[i]).index(1.0)
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("\nNumber Of Images Tested =", all_count)
print("Model Accuracy =", (correct_count / all_count) * 100, "%")