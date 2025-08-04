import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import RegressionModel

data = pd.read_csv('data/2017.csv')
x_data = data[['Economy..GDP.per.Capita.','Freedom']].to_numpy()
y_data = data[['Happiness.Score']].to_numpy()

model = RegressionModel(x_data, y_data, 0)

model.train(0.01,5000)


