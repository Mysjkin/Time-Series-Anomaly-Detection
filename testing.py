import json
from keras.models import model_from_json

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