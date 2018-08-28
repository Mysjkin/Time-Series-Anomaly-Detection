
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./data/100.csv', nrows=2500)

plt.plot(range(0,2100), df[0:2100]['v5'], color='blue')
plt.show()
