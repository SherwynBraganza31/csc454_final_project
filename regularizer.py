import pandas as pd
import numpy as np

df_original = pd.read_csv('./cleaned_datasets/test_outlier_removed.csv').iloc[2:,0:15]
df_new = pd.DataFrame(columns=range(0,11))
df_original = df_original.replace(np.nan,float(0))

for idx, x in df_original.iterrows():
    col = int(float(x.iloc[-1]))
    df_new = df_new.append({col: 1}, ignore_index=True)

df_original = df_original.iloc[:,0:-1]
df_new = df_new.replace(np.nan,float(0))
df_new.to_csv('./regularized_datasets/regularized_test_labels2.csv')
df_original.to_csv('./regularized_datasets/regularized_test2.csv')

