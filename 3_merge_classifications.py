import os
from datetime import datetime

import numpy as np
import pandas as pd


def merge(df1, df2):
    types = ['Type_1', 'Type_2', 'Type_3']
    ll = min(df1.shape[0], df2.shape[0])
    df_out = pd.DataFrame(columns=df1.columns)
    for i in range(ll):
        proba1 = df1.loc[i, types]
        proba2 = df2.loc[i, types]
        proba = proba1 if np.max(proba1) > np.max(proba2) else proba2
        df_out.loc[i, :] = (df1.loc[i, 'image_name'], ) + tuple(proba)
    return df_out


df1 = pd.read_csv("results/submission_mean_mixed_cnn.csv")
df2 = pd.read_csv("results/submission_mean_cv=6_squeezenet.csv")

df3 = merge(df1, df2)

# Crop between [0.03, 0.97]
for t in ['Type_1', 'Type_2', 'Type_3']:
    mask = df3[t] < 0.03
    df3.loc[mask, t] = 0.03
    mask = df3[t] > 0.97
    df3.loc[mask, t] = 0.97

info = 'final_classification'

now = datetime.now()
sub_file = 'submission_' + info + '.csv'
sub_file = os.path.join('results', sub_file)
df3.to_csv(sub_file, index=False)

df4 = pd.read_csv("results/stage1_submission_final_classification.csv")
df_final = pd.concat([df3, df4])

now = datetime.now()
sub_file = 'stage12_submission_' + info + '.csv'
sub_file = os.path.join('results', sub_file)
df_final.to_csv(sub_file, index=False)
