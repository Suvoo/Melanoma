import os
import pandas as pd
from sklearn import model_selection

input_path = 'train.csv'
df = pd.read_csv(input_path)
df['kfold'] = -1
df = df.sample(frac = 1).reset_index(drop = True) #shuffling the data, and reset index
y = df.target.values
print(len(y))
kf = model_selection.StratifiedKFold(n_splits=5)
for fold_,(train_idx, test_idx) in enumerate(kf.split(X=df,y=y)):
    df.loc[test_idx,'kfold'] = fold_
df.to_csv('train_folds.csv')

