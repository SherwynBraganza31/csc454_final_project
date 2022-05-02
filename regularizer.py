import pandas as pd
import numpy as np

df_original = pd.read_csv('./original_datasets/original_train.csv').iloc[:,2:]
df_new = pd.DataFrame(columns=range(0,11))
df_original = df_original.replace(np.nan,float(0))

for idx, x in df_original.iterrows():
    df_new = df_new.append({x.iloc[-1] : 1}, ignore_index=True)

df_original = df_original.iloc[:,0:-1]
df_new = df_new.replace(np.nan,float(0))
df_new.to_csv('./regularized_datasets/regularized_train_labels.csv')
df_original.to_csv('/regularized_datasets/regularized_train.csv')

