import pandas as pd
import numpy as np

df_train= pd.read_csv('./cleaned_datasets/train.csv').iloc[2:,0:15]
df_test= pd.read_csv('./cleaned_datasets/test.csv').iloc[2:,0:15]
df_outliers = pd.read_csv('./cleaned_datasets/outliers.csv').iloc[2:,0:15]


df_train_labels = pd.DataFrame(columns=range(0,11))
df_test_labels = pd.DataFrame(columns=range(0,11))
df_train = df_train.replace(np.nan,float(0))
df_test = df_test.replace(np.nan,float(0))

for idx, x in df_train.iterrows():
    col = int(float(x.iloc[-1]))
    df_train_labels = df_train_labels.append({col: 1}, ignore_index=True)

for idx, x in df_test.iterrows():
    col = int(float(x.iloc[-1]))
    df_test_labels = df_test_labels.append({col: 1}, ignore_index=True)

df_train = df_train.iloc[:,0:-1]
df_test = df_test.iloc[:,0:-1]

df_train_labels = df_train_labels.replace(np.nan,float(0))
df_test_labels = df_test_labels.replace(np.nan,float(0))

df_train_labels.to_csv('./regularized_datasets/regularized_train_labels.csv')
df_test_labels.to_csv('./regularized_datasets/regularized_test_labels.csv')
df_train.to_csv('./regularized_datasets/regularized_train.csv')
df_test.to_csv('./regularized_datasets/regularized_test.csv')

