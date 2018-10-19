import json
from keras.models import model_from_json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import preprocessing
import graphs

def reshape_input_and_decoded(input_tensor, decoded):
    input_flat = input_tensor.reshape((samples * segment_size, num_features))     
    decoded_flat = decoded.reshape((samples * segment_size, num_features))

    return input_flat, decoded_flat

def reconstruction_error(input_tensor, decoded):
    input_flat, decoded_flat = reshape_input_and_decoded(input_tensor, decoded)
    return np.sum(np.abs(input_flat - decoded_flat), axis=1)

def load_model_and_parameters():
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

    return model, parameters

def load_data(parameters):
    # Load data.
    df = pd.read_csv('./data/100.csv', nrows=parameters['numSamples'], skiprows=(1, 30000))
    # Remove elapsed time - assuming time is increased by one unit.
    df = df[['v5','mlii']]
    return df

def normalize_data(data):
    # Normalize data.
    df = data.values.astype('float32')
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

def get_training_test_data(data, parameters):
    # Split data into test and training.
    test_num = int(data.shape[0] * parameters['testSplit'])
    test_df = data[0:test_num]
    training_df = data[test_num:]
    return training_df, test_df


model, parameters = load_model_and_parameters()

data = load_data(parameters)
data_normalized = normalize_data(data)

_, test_df = get_training_test_data(data, parameters)

input_tensor = preprocessing.create_input_tensor(test_df, parameters['segmentSize'])

decoded = model.predict(input_tensor)

segment_size = parameters['segmentSize']
num_features = test_df.shape[-1]
samples = input_tensor.shape[0]

error = reconstruction_error(input_tensor, decoded)

import matplotlib.pyplot as plt

input_flat, decoded_flat = reshape_input_and_decoded(input_tensor, decoded)

plt.plot(error[2000:2500], label='error')
plt.plot(input_flat[2000:2500,0], label='actual')
plt.plot(decoded_flat[2000:2500,0], label='predicted')
plt.legend()
plt.show()
