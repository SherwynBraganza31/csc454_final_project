import torch
from torch.utils.data import Dataset
from model import DatasetGen, CNN
import pandas

df_test = pandas.read_csv('./regularized_datasets/regularized_test.csv').iloc[:,1:]
df_test_labels = pandas.read_csv('./regularized_datasets/regularized_test_labels.csv').iloc[:,1:]

batch = 4
test_loader = torch.utils.data.DataLoader(DatasetGen([df_test, df_test_labels]),batch_size=batch,shuffle=True)

model_cnn = CNN()

model_cnn.load_state_dict(torch.load("music_recognition_cnn.sav"))
model_cnn.eval()

#music_params, labels = next(iter(test_loader))
model_cnn.eval()
accuracy = 0
count = 0

for music_params, labels in test_loader:
    with torch.no_grad():
        music_params = torch.reshape(music_params, [batch,1,1,14])
        test_output = model_cnn(music_params)

    pred_label = torch.max(test_output,1)[1]
    actual_label = torch.max(labels,1)[1]
    accuracy = (accuracy + (pred_label == actual_label).sum().item() / float(labels.size(0)))/2
    count = count +1

print("\nNumber Of Images Tested =", count*batch)
print("Model Accuracy =", accuracy * 100, "%")
