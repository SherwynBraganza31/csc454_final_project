import torch
from torch.utils.data import Dataset
from model import MLP_ANN, DatasetGen
import pandas

df_test = pandas.read_csv('./regularized_datasets/regularized_test.csv').iloc[:,1:]
df_test_labels = pandas.read_csv('./regularized_datasets/regularized_test_labels.csv').iloc[:,1:]

test_loader=torch.utils.data.DataLoader(DatasetGen([df_test, df_test_labels]),batch_size=256,shuffle=False)

model_ann = MLP_ANN()

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