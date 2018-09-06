
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import models
import preprocessing
import figure_creator

from keras.models import load_model
from keras import optimizers

df = pd.read_csv('./data/100.csv', nrows=2500)
df = df[['v5','mlii']]

df = df.values.astype('float32')
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

adam = optimizers.Adam()
model = models.get_model(2,4,5,adam)

input_tensor = preprocessing.create_input_tensor(df, 5)

model.fit(input_tensor, input_tensor, epochs=5, validation_split=0.2)

decoded = model.predict(input_tensor)

gc = figure_creator.GraphCreator()
gc.segmentedDataPlot(input_tensor, 5, 1)
gc.segmentedDataPlot(decoded, 5, 1)
plt.show()









