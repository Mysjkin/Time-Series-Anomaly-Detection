
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import models
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('./data/100.csv', nrows=2500)
df = df[['v5','mlii']]

df = df.as_matrix().astype('float32')
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

model = models.get_model(2,4,5,'Adam')




