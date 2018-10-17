import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys

# Not used currently.
""" def animatedDataSequence(data):
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    x = np.arange(0, 20, 0.1)
    ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
    line, = ax.plot(x, x - 5, 'r-', linewidth=2)

    def update(i):
        label = 'timestep {}'.format(i)
        line.set_ydata(x-5+i)
        ax.set_xlabel(label)
        return line, ax
    anim = FuncAnimation(fig, update, frames=np.arange(0,10), interval=200)
    anim.save('line.gif', writer='imagemagick', fps=30) """

def rangedSegmentedDataPlot(data, segment_size, label, feature_index=0, realSampleRange=(0,1)):
    startTensorElement = realSampleRange[0] 
    endTensorElement = realSampleRange[1] // segment_size
    segmentedDataPlot(data[startTensorElement:endTensorElement], segment_size, label, feature_index=feature_index)

def segmentedDataPlot(data, segment_size, label, feature_index=0):
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
    
        plt.plot(x,y, label=label)

