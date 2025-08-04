import numpy as np
import pandas as pd
from Regression import RegressionModel

data = pd.read_csv('data/non-linear.csv')
x = data['x'].to_numpy().reshape((data.shape[0], 1))
y = data['y'].to_numpy().reshape((data.shape[0], 1))

linear_regression = RegressionModel(x, y, 15,True)

linear_regression.train(0.01,50000)

