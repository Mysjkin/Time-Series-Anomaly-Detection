import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def segmentedDataPlot(data, segment_size, feature_index=0):
        """ Creates a plot of an tensor used as input/output from the
            LSTM-based model.

            :: 
        """
        samples = data.shape[0]
        x = range(0,segment_size*samples)
        y = np.zeros(segment_size*samples)

        k = 0
        for i in range(samples):
            for w in range(segment_size):
                y[k] = data[i][w][feature_index]
                k = k+1
    
        plt.plot(x,y)