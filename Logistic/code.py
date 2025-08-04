import numpy as np
import pandas as pd
from Logistic import LogisticModel

data = pd.read_csv('iris.csv')

x_axis = "petal_length"
y_axis = "petal_width"
iris_types = ['SETOSA', 'NOT_SETOSA']

x_train = data[[x_axis, y_axis]].to_numpy().reshape((-1, 2))
y_train = (data['class'] == "SETOSA").astype(int).to_numpy().reshape((-1, 1))

model = LogisticModel(x_train,y_train,0,True)
model.train(0.05,1000)

predictions = model.predict(x_train)
accuracy = np.mean(predictions == y_train.flatten())
print(f"Accuracy: {accuracy * 100:.2f}%")



