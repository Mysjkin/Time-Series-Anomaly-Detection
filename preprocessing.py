
import pandas as pd
import numpy as np

def create_input_tensor(data, segment_size):
    """ Creates an input tensor by converting a list of samples into
    a list of samples where each sample is a collection of timesteps
    equal to the segment_size.

    Tensor shape: (samples, timesteps, features).
    """
    num_features = data.shape[-1]
    num_samples = data.shape[0]
    num_samples = num_samples//segment_size
    input_tensor = np.zeros((num_samples, segment_size, num_features))
    for i in range(0, num_samples):
        input_tensor[i] = data[i*segment_size:i*segment_size+segment_size]
    return input_tensor
