
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import json

import models
import preprocessing
import graphs

from keras.models import load_model

# Load model parameters.
with open("modelParameters.json", "r") as model_param:
    parameters = json.load(model_param)

# Load data.
df = pd.read_csv('./data/100.csv', nrows=parameters['numSamples'])

""" Remove time stamps - under the assumption that
the distance between time stamps can be regarded as
one unit of time. This may not necessarily be the case,
however, for this small example it is fine. 
"""
df = df[['v5','mlii']]

# Normalize data.
df = df.values.astype('float32')
scaler = MinMaxScaler()
df = scaler.fit_transform(df)

# Split data into test and training.
test_num = int(df.shape[0] * parameters['testSplit'])
test_df = df[0:test_num]
training_df = df[test_num:]

# Create and compile keras model.
model = models.get_model(parameters['inputDimension'],
                        parameters['latentDimension'],
                        parameters['segmentSize'],
                        parameters['optimizer'],
                        parameters['learningRate'],
                        parameters['lossFunction'])

input_tensor = preprocessing.create_input_tensor(training_df, parameters['segmentSize'])

# Train model.
model.fit(input_tensor, input_tensor, epochs=parameters['epochs'], validation_split=parameters['validationSplit'])

# Save model and parameters.
model_dir = './model/'
model_name = 'epochs-{}-seg-{}-lat-{}'.format(parameters['epochs'], parameters['segmentSize'], parameters['latentDimension'])
model_json = model.to_json()
with open(model_dir+model_name+".json", "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_dir+model_name+".h5")

with open(model_dir+model_name+"Parameters.json", "w") as write_file:
    json.dump(parameters, write_file)










