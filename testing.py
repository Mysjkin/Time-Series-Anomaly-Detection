import json
from keras.models import model_from_json
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import preprocessing
import graphs

def reconstruction_error(input_tensor, decoded):
    input_tensors = input_tensor.reshape((samples * segment_size, num_features))     
    decoded = decoded.reshape((samples * segment_size, num_features))

    return np.sum(np.abs(input_tensors - decoded), axis=1)

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
df = pd.read_csv('./data/100.csv', nrows=parameters['numSamples'], skiprows=(1, 30000))
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

segment_size = parameters['segmentSize']
num_features = test_df.shape[-1]
samples = input_tensor.shape[0]

range_to_display = (0,500)

graphs.rangedSegmentedDataPlot(input_tensor, parameters['segmentSize'], 'Actual', 0, range_to_display)
graphs.rangedSegmentedDataPlot(decoded, parameters['segmentSize'], 'Predicted', 0, range_to_display)

error = reconstruction_error(input_tensor, decoded)

import matplotlib.pyplot as plt

print(error.shape)

plt.plot(range(0, 500), error[0:500,])

plt.show()

""" plt.ylabel('V5')
plt.xlabel('Time')
plt.title('Actual and predicted ECG data')
plt.legend()
plt.savefig('./output/figures/actual_predicted.png') """