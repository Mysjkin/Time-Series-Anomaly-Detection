import pandas as pd
import matplotlib.pyplot as plt

class PlotInfo:
    def __init__(self, name):
        self.figureName = name

class GraphCreator:
    """ Class for creating figures and graphs. 
    """
    
    def basicPlot(self, data, attributes):
        """ Creates a simple plot.

        :param data: data to plot.
        :type data: pandas dataframe or numpy array.
        :param attributes: information used to create the plot.
        :type attributes: PlotInfo class.
        """
        plt.plot(range(0,100), data[0,100])
        plt.show()
    
if __name__ == "__main__":

    gc = GraphCreator()