import json
from keras.models import model_from_json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import preprocessing
import graphs

model_dir = './model/'
# Load model parameters.
with open("modelParameters.json", "r") as model_param:
    parameters = json.load(model_param)
model_name = 'epochs-{}-seg-{}-lat-{}'.format(parameters['epochs'], parameters['segmentSize'], parameters['latentDimension'])

model_path = model_dir+model_name

with open(model_path+'.json', 'r') as model_arch:
    model_architecture = model_arch.read()

model = model_from_json(model_architecture)
model.load_weights(model_path+'.h5')

print(model)

# Load data.
df = pd.read_csv('./data/100.csv', nrows=parameters['numSamples'])
# Remove elapsed time - assuming time is increased by one unit.
df = df[['v5','mlii']]

# Normalize data.
df = df.values.astype('float32')
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

# Split data into test and training.
test_num = int(df.shape[0] * parameters['testSplit'])
test_df = df[0:test_num]
training_df = df[test_num:]

input_tensor = preprocessing.create_input_tensor(test_df, parameters['segmentSize'])

decoded = model.predict(input_tensor)

range_to_display = (0,250)

graphs.rangedSegmentedDataPlot(decoded, parameters['segmentSize'], 0, range_to_display)
graphs.rangedSegmentedDataPlot(input_tensor, parameters['segmentSize'], 0, range_to_display)

import matplotlib.pyplot as plt

plt.show()