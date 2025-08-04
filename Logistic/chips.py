import numpy as np
import pandas as pd
from Logistic import LogisticModel

data = pd.read_csv('data/microchips.csv')

x_axis = 'param_1'
y_axis = 'param_2'

x_train = data[[x_axis, y_axis]].to_numpy().reshape((-1, 2))
y_train = data['validity'].to_numpy().reshape((-1, 1))

model = LogisticModel(x_train,y_train,15,True)
model.train()

predictions = model.predict(x_train)
accuracy = np.mean(predictions == y_train.flatten())
print(f"Accuracy: {accuracy * 100:.2f}%")
